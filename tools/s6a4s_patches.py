"""S6A4: bounded, one-segment-per-CTA attention. CPU-only generator.

Only the two named lab outputs are written. No imports of a layer, CUDA, or torch.
HYA helpers retain the base's Apache-2.0 attribution. --check proves that removing
marked additions (with both selectors off) recovers the exact base bytes.
Decode retains independent FP16 MMA chains with the design note's 32/128 role caps
(launch 80, 3 CTAs/SM). Verify (BK=32) needs 80 consumer GPRs, so it runs 2 CTAs/SM with
9 stages and launch 80 / producer 32 / consumer 104 (see GEOM below and sonnet_a4_report.md).
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_tinys7_c1.py"
LAB = ROOT / "kernels/lab/k3_s6a4s_ab.py"
SHIP = ROOT / "kernels/lab/k3_s6a4s_ship.py"
BASE_SHA256 = "07eb81e0a91a5161e23e0f794abe1f536788ff4200326c209c484802092c6e10"
BEGIN = "# S6A4 BEGIN "
END = "# S6A4 END "


def once(text, old, new):
    assert text.count(old) == 1, ("anchor", old[:100], text.count(old))
    return text.replace(old, new, 1)


def between(text, start, end):
    assert text.count(start) == 1, start
    a = text.index(start)
    b = text.index(end, a)
    return text[a:b]


def literal(text, name):
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(nodes) == 1, name
    return ast.literal_eval(nodes[0].value)


# S6A4S residency redesign (measured with ptxas on the original source, see
# notes/study/evo/s6/sonnet_a4_report.md): at BK=32 the verify consumers need 80 GPRs/thread and the
# producers ~30, which cannot fit three 384-thread CTAs/SM (56-register launch, 24/72 roles: pool bound
# gives consumers <=72 only with a producer that spills).  Verify therefore runs 2 CTAs/SM with 9 stages
# (same 144 KiB KV ring per SM), launch 80 / producer 32 / consumer 104 (pool 30720 = 384*80, exact).
# Decode keeps the design's 256-thread, 3 CTAs/SM x 6 stages with the design's 32/128 role caps (launch 80).
GEOM = {8: dict(ST=9, NT=384, MINB=2, RINIT=80, PREG=32, CREG=104, SEG2=66),
        1: dict(ST=6, NT=256, MINB=3, RINIT=80, PREG=32, CREG=128, SEG2=99)}
SM_SMEM_BYTES = 233472   # sm_90a: 228 KiB per SM; each CTA also reserves 1 KiB
SM_REGS = 65536


def seg_count(B, L):
    """Segments per call: one grid wave = MINB*132 CTAs; two waves at B=16 (G = B*SEG2/2)."""
    return B * GEOM[L]["SEG2"] // 2


def smem_cap(L):
    """Largest per-CTA shared footprint (dynamic + static, INCLUDING the 1 KiB the hardware reserves per CTA,
    which cuobjdump reports as SHARED:1024 even for kernels without shared memory) that still leaves MINB
    resident CTAs; the three-CTA decode keeps the design's 72 KiB/CTA budget."""
    cap = SM_SMEM_BYTES // GEOM[L]["MINB"]
    return min(cap, 72 * 1024) if GEOM[L]["MINB"] >= 3 else cap


def geom_text():
    rows = []
    for key in ("ST", "NT", "MINB", "RINIT", "PREG", "CREG", "SEG2"):
        rows.append("  static constexpr int %s = L==8 ? %d : %d;" % (key.ljust(5), GEOM[8][key], GEOM[1][key]))
    return "\n".join(rows)


CUDA_T = r'''
// S6A4: HYA-derived accessors/fragment layouts; independent retirement, no claims.
constexpr int S6_BK=32, S6_SB=4096;
// Per-instantiation geometry (verify L=8 / decode L=1); mirrored by tools/s6a4s_patches.py GEOM.
template<int L> struct S6C {
@GEOM@
  static_assert(NT==(L==8?384:256) && PREG%8==0 && CREG%8==0 && RINIT%8==0, "S6A4 geometry");
  // Role registers must exactly tile the CTA register pool allocated at launch.
  static_assert(128*PREG+(NT-128)*CREG==NT*RINIT, "S6A4 register pool");
  static_assert(NT*RINIT*MINB<=65536, "S6A4 residency registers");
};
struct S6Params {
  Params h;
  uint8_t* q;
  int4* desc;
  int2* bounds;
  float* partial;
  float2* ml;
  int segments;
};
template<int ST> struct alignas(1024) S6Sh {
  alignas(1024) uint8_t k[ST*S6_SB], v[ST*S6_SB];
  alignas(1024) uint8_t q[8192];
  // Two 32-key, 72-column V^T tiles per head. 32-byte swizzle.
  alignas(1024) uint8_t vt[2][2][2304];
  int idx[ST][32];
  uint64_t full[ST], empty[ST], vtf[2], vte[2];
};
template<int L> using S6Shared = S6Sh<S6C<L>::ST>;
// Both instantiations must leave MINB CTAs resident (each CTA also reserves 1 KiB of shared memory).
static_assert(sizeof(S6Shared<8>)+1024<=233472/S6C<8>::MINB, "S6A4 verify shared-memory gate");
static_assert(sizeof(S6Shared<1>)+1024<=233472/S6C<1>::MINB, "S6A4 decode shared-memory gate");
__device__ __forceinline__ uint64_t s6_desc32(uint32_t addr) {
  return (uint64_t)((addr&0x3ffff)>>4) | ((uint64_t)1<<16)
       | ((uint64_t)(256>>4)<<32) | ((uint64_t)3<<62);
}
__device__ __forceinline__ void s6_qk(float* d,uint64_t qa,uint64_t kb,int add) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%18,0;\n"
    "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 "
    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},%16,%17,p,1,1;\n}\n"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),
      "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15])
    : "l"(qa),"l"(kb),"r"(add));
}
__device__ __forceinline__ uint64_t s6_first() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ void s6_copy(uint32_t dst,const void* src,int valid,uint64_t pol) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0],[%1],16,%2,%3;\n"
      ::"r"(dst),"l"(src),"r"(valid),"l"(pol):"memory");
}
__device__ __forceinline__ uint64_t s6_last() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ void s6_keep(float* p,float x,uint64_t pol) {
  asm volatile("st.global.L1::evict_last.L2::cache_hint.f32 [%0],%1,%2;" ::"l"(p),"f"(x),"l"(pol):"memory");
}
__device__ __forceinline__ void s6_keep_ml(float2* p,float m,float l,uint64_t pol) {
  s6_keep(reinterpret_cast<float*>(p),m,pol);s6_keep(reinterpret_cast<float*>(p)+1,l,pol);
}

// DESCRIPTORS: fixed grid, length-weighted split counts, exclusive slots.
// Empty descriptors are explicit and are skipped by BOTH main and merge.
template<int L>
__global__ void __launch_bounds__(128, 9) s6a4s_pre(const __grid_constant__ S6Params a) {
  const Params& p=a.h;
  const int tid=threadIdx.x, b=blockIdx.x/L, j=blockIdx.x%L;
  pdl_wait();
  if (blockIdx.x==0 && tid<32) {
    int tiles=tid<p.B ? (__ldg(p.seq_lens+tid)+L+31)/32 : 0;
    int prefix=tiles;
    #pragma unroll
    for (int d=1;d<32;d*=2) {
      int t=__shfl_up_sync(0xffffffffu,prefix,d);
      if (tid>=d) prefix+=t;
    }
    const int total=__shfl_sync(0xffffffffu,prefix,31);
    if (tid<p.B) {
      const int before=prefix-tiles, extra=a.segments-p.B;
      const int first=tid+(int)((long long)before*extra/total);
      const int count=1+(int)((long long)prefix*extra/total)-(int)((long long)before*extra/total);
      a.bounds[tid]=make_int2(first,count);
      const int seq=__ldg(p.seq_lens+tid);
      for (int s=0;s<count;++s) {
        const int lo=(int)((long long)tiles*s/count)*32;
        const int hi=min(seq+L,(int)((long long)tiles*(s+1)/count)*32);
        a.desc[first+s]=make_int4(tid,lo,hi,seq);
      }
    }
  }
  // Same eight-lane normalization and FP8 / scaled FP16 query path as HYA.
  const int hh=tid>>3, part=tid&7;
  const uint4 qv=*reinterpret_cast<const uint4*>(p.qkv+(ix_t)(b*L+j)*p.ld_qkv+hh*64+part*8);
  const __nv_bfloat162* h2=reinterpret_cast<const __nv_bfloat162*>(&qv);
  float x[8];
  #pragma unroll
  for (int e=0;e<4;++e) { float2 f=__bfloat1622float2(h2[e]);x[2*e]=f.x;x[2*e+1]=f.y; }
  float ss=0.f;
  #pragma unroll
  for (int e=0;e<8;++e) ss+=x[e]*x[e];
  float r=rsqrtf(oct_sum(ss)*(1.f/64.f)+1e-5f);
  #pragma unroll
  for (int e=0;e<8;++e) x[e]*=r;
  // Selected full-attention shapes have no RoPE; unsupported configs fall back.
  if constexpr (L==8) {
    const int rr=j*8+(hh&7), kb=part*8;
    uint2 v=make_uint2(pack_e4m3x4(pack_h2(x[0],x[1]),pack_h2(x[2],x[3])),
                      pack_e4m3x4(pack_h2(x[4],x[5]),pack_h2(x[6],x[7])));
    *reinterpret_cast<uint2*>(a.q+b*8192+(hh>>3)*4096+rr*64+(((kb>>4)^((rr>>1)&3))<<4)+(kb&15))=v;
  } else {
    const float sc=p.scale_log2*__ldg(p.k_scale);
    // HYA's decode Q is FP16 (higher precision than the permitted E4M3 minimum).
    uint4 v=make_uint4(pack_h2(x[0]*sc,x[1]*sc),pack_h2(x[4]*sc,x[5]*sc),
                      pack_h2(x[2]*sc,x[3]*sc),pack_h2(x[6]*sc,x[7]*sc));
    *reinterpret_cast<uint4*>(a.q+b*8192+hh*128+part*16)=v;
  }
  if (tid<32) {
    const int rr=tid>>3, kind=rr/2, kh=rr&1;
    const int col=(kind==0?p.Hq:p.Hq+p.Hkv)*64+kh*64+part*8;
    const uint4 v=*reinterpret_cast<const uint4*>(p.qkv+(ix_t)(b*L+j)*p.ld_qkv+col);
    const __nv_bfloat162* h=reinterpret_cast<const __nv_bfloat162*>(&v);
    float z[8], sum=0.f;
    #pragma unroll
    for (int e=0;e<4;++e) { float2 f=__bfloat1622float2(h[e]);z[2*e]=f.x;z[2*e+1]=f.y; }
    #pragma unroll
    for (int e=0;e<8;++e) sum+=z[e]*z[e];
    sum=oct_sum(sum);
    const float norm=kind==0?rsqrtf(sum*(1.f/64.f)+1e-5f):1.f;
    const float sc=__fdividef(1.f,__ldg(kind==0?p.k_scale:p.v_scale));
    uint32_t w0=0,w1=0;
    #pragma unroll
    for (int e=0;e<4;++e) {
      w0|=(uint32_t)to_e4m3((z[e]*norm)*sc)<<(8*e);
      w1|=(uint32_t)to_e4m3((z[4+e]*norm)*sc)<<(8*e);
    }
    const int64_t row=__ldg(p.cache_loc+b*L+j);
    if (row>=0) {
      uint8_t* dst=const_cast<uint8_t*>(cache_row(kind==0?p.k_cache:p.v_cache,p,row,kh*64+part*8));
      *reinterpret_cast<uint2*>(dst)=make_uint2(w0,w1);
    }
  }
  pdl_launch_dependents();
}

// Exactly 32 page indices, staged one tile ahead; no cache-derived host values.
template<int L>
__device__ __forceinline__ void s6_pages(const S6Params& a,S6Shared<L>& S,int4 d,int tile,int tid) {
  const Params& p=a.h;
  if (tid<32) {
    const int pos=d.y+32*tile+tid;
    S.idx[tile%S6C<L>::ST][tid]=(pos<min(d.z,d.w))?__ldg(p.page_table+(ix_t)d.x*p.max_pages+(pos>>p.lp)):0;
  }
  bar_sync_n(2,128);
}
template<int L>
__device__ __forceinline__ void s6_gather(const S6Params& a,S6Shared<L>& S,int4 d,int tile,int tid,uint64_t pol) {
  const Params& p=a.h;
  const int st=tile%S6C<L>::ST, ch=tid&7, r0=tid>>3;
  #pragma unroll
  for (int k=0;k<2;++k) {
    const int r=r0+16*k, pos=d.y+tile*32+r;
    const int64_t row=pos<d.w?(((int64_t)S.idx[st][r]<<p.lp)+(pos&(p.P-1))):
        (pos<d.z?__ldg(p.cache_loc+d.x*L+(pos-d.w)):0);
    const int so=r*128+((ch^(r&7))<<4);
    const uint8_t* kp=pos<d.z?cache_row(p.k_cache,p,row,ch*16):p.k_cache;
    const uint8_t* vp=pos<d.z?cache_row(p.v_cache,p,row,ch*16):p.v_cache;
    s6_copy(smem_u32(S.k+st*4096+so),kp,pos<d.z?16:0,pol);
    s6_copy(smem_u32(S.v+st*4096+so),vp,pos<d.z?16:0,pol);
  }
  cp_async_mbar_arrive(&S.full[st]);
}

// HYA's ldsm/prmt transpose, compressed from 64-key sw64 to 32-key sw32 (verify only).
__device__ __forceinline__ void s6_transpose(S6Shared<8>& S,int tile,int tid) {
  const int lane=tid&31, warp=tid>>5, hk=warp>>1, wq=warp&1;
  const int vw=lane>>2,vq=lane&3,vr0=(vw>=4)?1:0;
  const uint32_t sel0=vw>=4?0x7531u:0x6420u, sel1=vw>=4?0x6420u:0x7531u;
  const int key=16*wq+8*((lane>>3)&1)+(lane&7);
  const uint32_t ld=key*128+(((4*hk+((lane>>3)>>1))^(lane&7))<<4);
  uint8_t* vt=S.vt[tile&1][hk];
  const uint32_t off0=64*vw+((wq^(vw&1))<<4)+4*vq+32*vr0;
  const uint32_t off1=64*vw+((wq^(vw&1))<<4)+4*vq+32*(1-vr0);
  #pragma unroll
  for (int u=0;u<2;++u) {
    uint32_t r0,r1,r2,r3;
    ldsm_x4_t(smem_u32(S.v+(tile%S6C<8>::ST)*4096)+(u?(ld^32u):ld),r0,r1,r2,r3);
    #pragma unroll
    for (int g2=0;g2<2;++g2) {
      const uint32_t R0=g2?r2:r0,R1=g2?r3:r1;
      *reinterpret_cast<uint32_t*>(vt+off0+512*(2*u+g2))=prmt(R0,R1,sel0);
      *reinterpret_cast<uint32_t*>(vt+off1+512*(2*u+g2))=prmt(R0,R1,sel1);
    }
  }
}

template<int L>
__device__ __forceinline__ void s6_producer(const S6Params& a,S6Shared<L>& S,int4 d) {
  constexpr int ST=S6C<L>::ST;
  const int tid=threadIdx.x, nt=(d.z-d.y+31)/32;
  const uint64_t pol=s6_first();
  #pragma unroll 1
  for (int t=0;t<min(ST,nt);++t) { s6_pages<L>(a,S,d,t,tid);s6_gather<L>(a,S,d,t,tid,pol); }
  for (int t=0;t<nt;++t) {
    if constexpr (L==8) {
      mbar_wait_full(&S.full[t%ST],(t/ST)&1);
      if (t>=2) mbar_wait_full(&S.vte[t&1],((t/2)-1)&1);
      s6_transpose(S,t,tid);
      fence_proxy_async();
      mbar_arrive(&S.vtf[t&1]);
    }
    if (t+ST<nt) {
      // Page-table dependency is advanced while consumers retire this tile.
      s6_pages<L>(a,S,d,t+ST,tid);
      mbar_wait_full(&S.empty[t%ST],(t/ST)&1);
      s6_gather<L>(a,S,d,t+ST,tid,pol);
    }
  }
  cp_wait<0>();
  // Terminal role; no consumer-side epilogue or full-CTA barrier after return.
}

__device__ __forceinline__ void s6_verify(const S6Params& a,S6Shared<8>& S,int4 d) {
  constexpr int ST=S6C<8>::ST;
  const int tid=threadIdx.x-128, lane=tid&31;
  const int warp=__shfl_sync(0xffffffffu,tid>>5,0),hk=warp>>2,wsub=warp&3;
  const int g=lane>>2, tig=lane&3;
  float o[32]={}, den[4]={}, m[2]={-INFINITY,-INFINITY};
  const float sc=a.h.scale_log2*__ldg(a.h.k_scale);
  const int nt=(d.z-d.y+31)/32;
  for (int t=0;t<nt;++t) {
    const int st=t%ST;
    mbar_wait_full(&S.full[st],(t/ST)&1);
    float s[16];
    wgmma_fence();
    {
      uint64_t qd=desc_sw64(smem_u32(S.q+hk*4096));
      uint64_t kd=desc_sw128(smem_u32(S.k+st*4096+hk*64));
      s6_qk(s,qd,kd,0);s6_qk(s,qd+2,kd+2,1);
    }
    wgmma_commit();wgmma_wait<0>();
    #pragma unroll
    for (int e=0;e<16;++e) reg_fence(s[e]);
    #pragma unroll
    for (int hf=0;hf<2;++hf) {
      const int hi=min(d.z-1,d.w+2*wsub+hf);
      #pragma unroll
      for (int n=0;n<4;++n)
        #pragma unroll
        for (int e=0;e<2;++e)
          if (d.y+t*32+8*n+2*tig+e>hi) s[4*n+2*hf+e]=-INFINITY;
    }
    // Same delayed-rescale FP8 probability path as HYA; one P fragment.
    // m is the exponential reference, not necessarily this tile's maximum.
    uint32_t pa[4];
    if (t==0) {
      float lo=-INFINITY,hi=-INFINITY;
      #pragma unroll
      for(int n=0;n<4;++n) {lo=fmaxf(lo,fmaxf(s[4*n],s[4*n+1]));hi=fmaxf(hi,fmaxf(s[4*n+2],s[4*n+3]));}
      m[0]=quad_max(lo)*sc;m[1]=quad_max(hi)*sc;
    }
    #pragma unroll 1
    for (int attempt=0;attempt<2;++attempt) {
      const float lo=m[0]==-INFINITY?0.f:m[0], hi=m[1]==-INFINITY?0.f:m[1];
      pa[0]=e4m3x4f(ex2(fmaf(s[0],sc,-lo)),ex2(fmaf(s[1],sc,-lo)),ex2(fmaf(s[4],sc,-lo)),ex2(fmaf(s[5],sc,-lo)));
      pa[1]=e4m3x4f(ex2(fmaf(s[2],sc,-hi)),ex2(fmaf(s[3],sc,-hi)),ex2(fmaf(s[6],sc,-hi)),ex2(fmaf(s[7],sc,-hi)));
      pa[2]=e4m3x4f(ex2(fmaf(s[8],sc,-lo)),ex2(fmaf(s[9],sc,-lo)),ex2(fmaf(s[12],sc,-lo)),ex2(fmaf(s[13],sc,-lo)));
      pa[3]=e4m3x4f(ex2(fmaf(s[10],sc,-hi)),ex2(fmaf(s[11],sc,-hi)),ex2(fmaf(s[14],sc,-hi)),ex2(fmaf(s[15],sc,-hi)));
      constexpr uint32_t PB=0x07070707u;
      const uint32_t ov=((pa[0]+PB)|(pa[1]+PB)|(pa[2]+PB)|(pa[3]+PB))&0x80808080u;
      if (attempt || !__any_sync(0xffffffffu,ov!=0u)) break;
      float ml=-INFINITY,mh=-INFINITY;
      #pragma unroll
      for(int n=0;n<4;++n) {ml=fmaxf(ml,fmaxf(s[4*n],s[4*n+1]));mh=fmaxf(mh,fmaxf(s[4*n+2],s[4*n+3]));}
      ml=fmaxf(m[0],quad_max(ml)*sc);mh=fmaxf(m[1],quad_max(mh)*sc);
      const float al=m[0]==-INFINITY?0.f:ex2(m[0]-ml), ah=m[1]==-INFINITY?0.f:ex2(m[1]-mh);
      #pragma unroll
      for(int n=0;n<8;++n) {o[4*n]*=al;o[4*n+1]*=al;o[4*n+2]*=ah;o[4*n+3]*=ah;}
      den[0]*=al;den[1]*=al;den[2]*=ah;den[3]*=ah;m[0]=ml;m[1]=mh;
    }
    mbar_wait_full(&S.vtf[t&1],(t/2)&1);
    mbar_arrive(&S.empty[st]);
    #pragma unroll
    for(int e=0;e<32;++e) reg_fence(o[e]);
    #pragma unroll
    for(int e=0;e<4;++e) {reg_fence(den[e]);asm volatile("" : "+r"(pa[e]));}
    wgmma_fence();
    wgmma_m64n72k32_e4m3(o,den,pa,s6_desc32(smem_u32(S.vt[t&1][hk])));
    wgmma_commit();wgmma_wait<0>();
    #pragma unroll
    for(int e=0;e<32;++e) reg_fence(o[e]);
    #pragma unroll
    for(int e=0;e<4;++e) {reg_fence(den[e]);asm volatile("" : "+r"(pa[e]));}
    mbar_arrive(&S.vte[t&1]);
  }
  const uint64_t pol=s6_last();
  #pragma unroll
  for(int hf=0;hf<2;++hf) {
    const int row=(2*wsub+hf)*16+hk*8+g;
    const int slot=blockIdx.x*128+row;
    #pragma unroll
    for(int n=0;n<8;++n) {
      s6_keep(a.partial+slot*64+8*n+2*tig,o[4*n+2*hf],pol);
      s6_keep(a.partial+slot*64+8*n+2*tig+1,o[4*n+2*hf+1],pol);
    }
    // WGMMA's eight all-one columns give duplicate denominator columns.
    if(tig==0) s6_keep_ml(a.ml+slot,m[hf],den[2*hf],pol);
  }
}

__device__ __forceinline__ void s6_decode(const S6Params& a,S6Shared<1>& S,int4 d) {
  constexpr int ST=S6C<1>::ST;
  const int tid=threadIdx.x-128,lane=tid&31,warp=__shfl_sync(0xffffffffu,tid>>5,0);
  const int hk=warp&1,sub=warp>>1,g=lane>>2,tig=lane&3;
  // Two independent 16-key chains per head; only eight live score values.
  uint32_t qa[4][2];
  {
    const uint8_t* qs=S.q+hk*1024;
    uint4 l0=*reinterpret_cast<const uint4*>(qs+g*128+32*tig);
    uint4 l1=*reinterpret_cast<const uint4*>(qs+g*128+32*tig+16);
    qa[0][0]=l0.x;qa[0][1]=l0.y;qa[1][0]=l0.z;qa[1][1]=l0.w;
    qa[2][0]=l1.x;qa[2][1]=l1.y;qa[3][0]=l1.z;qa[3][1]=l1.w;
  }
  float o[8][2]={}, m=-INFINITY,l=0.f;
  for(int t=0;t<(d.z-d.y+31)/32;++t) {
    const int st=t%ST;
    mbar_wait_full(&S.full[st],(t/ST)&1);
    float s[2][2];
    #pragma unroll
    for(int n=0;n<2;++n) {
      const int key=16*sub+8*n+g;
      uint4 kb=lds128(S.k+st*4096+key*128+(((hk*4+tig)^(key&7))<<4));
      uint32_t kw[4]={kb.x,kb.y,kb.z,kb.w};
      float se[4]={},so[4]={};
      #pragma unroll
      for(int h=0;h<2;++h) {
        mma_f16(se,qa[2*h][0],qa[2*h][1],qa[2*h+1][0],qa[2*h+1][1],cvt2(kw[2*h]),cvt2h(kw[2*h]));
        mma_f16(so,qa[2*h][0],qa[2*h][1],qa[2*h+1][0],qa[2*h+1][1],cvt2(kw[2*h+1]),cvt2h(kw[2*h+1]));
      }
      s[n][0]=se[0]+so[2];s[n][1]=se[1]+so[3];
      #pragma unroll
      for(int e=0;e<2;++e) if(d.y+t*32+16*sub+8*n+2*tig+e>=d.z) s[n][e]=-INFINITY;
    }
    const float mx=quad_max(fmaxf(fmaxf(s[0][0],s[0][1]),fmaxf(s[1][0],s[1][1])));
    const float mn=fmaxf(m,mx);
    if(mn!=-INFINITY) {
      const float al=m==-INFINITY?0.f:ex2(m-mn);m=mn;
      float ls=0.f;
      #pragma unroll
      for(int n=0;n<2;++n)
        #pragma unroll
        for(int e=0;e<2;++e) {s[n][e]=ex2(s[n][e]-mn);ls+=s[n][e];}
      l=l*al+ls;
      #pragma unroll
      for(int n=0;n<8;++n) {o[n][0]*=al;o[n][1]*=al;}
      uint32_t b0[8],b1[8];
      load_v_frags(S.v+st*4096,16*sub,hk,g,tig,b0,b1);
      const uint32_t a0=pack_h2(s[0][0],s[0][1]),a2=pack_h2(s[1][0],s[1][1]);
      #pragma unroll
      for(int n=0;n<8;++n) {
        // The unused lower eight query rows have zero A fragments. Do not
        // carry their accumulator registers between tiles (two live outputs).
        float acc[4]={o[n][0],o[n][1],0.f,0.f};
        mma_f16(acc,a0,0u,a2,0u,b0[n],b1[n]);o[n][0]=acc[0];o[n][1]=acc[1];
      }
    }
    __syncwarp();mbar_arrive(&S.empty[st]);
  }
  // The second warp chain writes only CTA-local scratch. Its companion owns
  // the exclusive global partial. All query loads retired before Q is reused.
  bar_sync_n(1,128);
  float* xs=reinterpret_cast<float*>(S.q)+((warp&1)*32+lane)*18;
  if(sub==1) {
    xs[0]=m;xs[1]=l;
    #pragma unroll
    for(int n=0;n<8;++n) {xs[2+2*n]=o[n][0];xs[3+2*n]=o[n][1];}
  }
  bar_sync_n(1,128);
  if(sub==0) {
    const uint64_t pol=s6_last();
    const float mm=fmaxf(m,xs[0]),a0=m==-INFINITY?0.f:ex2(m-mm),a1=xs[0]==-INFINITY?0.f:ex2(xs[0]-mm);
    const int slot=blockIdx.x*16+hk*8+g;
    const float ls=quad_sum(l*a0+xs[1]*a1);
    #pragma unroll
    for(int n=0;n<8;++n) {
      s6_keep(a.partial+slot*64+16*tig+n,o[n][0]*a0+xs[2+2*n]*a1,pol);
      s6_keep(a.partial+slot*64+16*tig+8+n,o[n][1]*a0+xs[3+2*n]*a1,pol);
    }
    if(tig==0) s6_keep_ml(a.ml+slot,mm,ls,pol);
  }
}

template<int L>
__global__ void __launch_bounds__(S6C<L>::NT,S6C<L>::MINB) s6a4s_segment(const __grid_constant__ S6Params a) {
  constexpr int ST=S6C<L>::ST, NTH=S6C<L>::NT;
  extern __shared__ __align__(1024) uint8_t raw[];
  S6Shared<L>& S=*reinterpret_cast<S6Shared<L>*>(raw);
  pdl_wait();
  const int4 loaded=a.desc[blockIdx.x];
  // HYA's lane-zero broadcasts prove the per-segment address math uniform.
  const int4 d=make_int4(__shfl_sync(0xffffffffu,loaded.x,0),__shfl_sync(0xffffffffu,loaded.y,0),
                        __shfl_sync(0xffffffffu,loaded.z,0),__shfl_sync(0xffffffffu,loaded.w,0));
  if(d.y>=d.z) {pdl_launch_dependents();return;}
  const int tid=threadIdx.x;
  if(tid==0) {
    for(int st=0;st<ST;++st) {mbar_init_n(&S.full[st],128);mbar_init_n(&S.empty[st],L==8?256:128);}
    for(int st=0;st<2;++st) {mbar_init_n(&S.vtf[st],128);mbar_init_n(&S.vte[st],256);}
    mbar_fence_init();
  }
  for(int v=tid;v<(L==8?512:128);v+=NTH)
    cp_async16(smem_u32(S.q+v*16),a.q+d.x*8192+v*16,16);
  if constexpr(L==8) {
    for(int v=tid;v<256;v+=NTH) {
      int buf=v/128,hk=(v/64)&1,col=v&63;
      *reinterpret_cast<uint32_t*>(S.vt[buf][hk]+2048+col*4)=0x38383838u;
    }
  }
  cp_commit();cp_wait<0>();fence_proxy_async();__syncthreads();
  if(tid<128) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(S6C<L>::PREG): "memory");
    s6_producer<L>(a,S,d);
    pdl_launch_dependents();return;
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(S6C<L>::CREG): "memory");
    if constexpr(L==8) s6_verify(a,S,d); else s6_decode(a,S,d);
    pdl_launch_dependents();return;
  }
}

template<int L>
__global__ void __launch_bounds__(128,9) s6a4s_merge(const __grid_constant__ S6Params a) {
  pdl_wait();
  const Params& p=a.h;
  const int tid=threadIdx.x,b=blockIdx.x/L,j=blockIdx.x%L,hh=tid>>3,part=tid&7;
  const int2 range=a.bounds[b];
  float m=-INFINITY;
  // Ascending exclusive slots, same order on every replay.
  for(int s=range.x;s<range.x+range.y;++s) {
    int4 d=a.desc[s];
    if(d.y<d.z) m=fmaxf(m,a.ml[s*(L*16)+j*16+hh].x);
  }
  float y[8]={},ls=0.f;
  for(int s=range.x;s<range.x+range.y;++s) {
    int4 d=a.desc[s];
    if(d.y>=d.z) continue;
    const int slot=s*(L*16)+j*16+hh;
    const float2 ml=a.ml[slot];
    if(ml.x==-INFINITY || ml.y==0.f) continue;
    const float scale=ex2(ml.x-m);
    ls=fmaf(ml.y,scale,ls);
    #pragma unroll
    for(int e=0;e<8;++e) y[e]=fmaf(a.partial[slot*64+part*8+e],scale,y[e]);
  }
  const float inv=ls>0.f?__fdividef(__ldg(p.v_scale),ls):0.f;
  float ss=0.f;
  #pragma unroll
  for(int e=0;e<8;++e) {
    // Stock _r1_attention: BF16 attention output BEFORE RMSNorm and gate.
    y[e]=__bfloat162float(__float2bfloat16_rn(y[e]*inv));ss+=y[e]*y[e];
  }
  const float rn=rsqrtf(oct_sum(ss)*(1.f/64.f)+1e-5f);
  const uint4 graw=*reinterpret_cast<const uint4*>(p.qkv+(ix_t)(b*L+j)*p.ld_qkv+(p.Hq+2*p.Hkv)*64+hh*64+part*8);
  const __nv_bfloat162* g2=reinterpret_cast<const __nv_bfloat162*>(&graw);
  uint4 ov;uint32_t* ow=reinterpret_cast<uint32_t*>(&ov);
  #pragma unroll
  for(int e=0;e<4;++e) {
    float2 gt=__bfloat1622float2(g2[e]);
    __nv_bfloat162 v=__floats2bfloat162_rn(y[2*e]*rn*sigm(gt.x),y[2*e+1]*rn*sigm(gt.y));
    ow[e]=*reinterpret_cast<uint32_t*>(&v);
  }
  *reinterpret_cast<uint4*>(p.out+(ix_t)(b*L+j)*(p.Hq*64)+hh*64+part*8)=ov;
  // No reset loop: every read slot was written in this call, or skipped by desc.
  pdl_launch_dependents();
}
} // anonymous namespace
'''

CUDA = CUDA_T.replace("@GEOM@", geom_text())
assert "@" not in CUDA

HOST = r'''
template<int L>
void s6_check() {
  auto k=s6a4s_segment<L>;
  TORCH_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sizeof(S6Shared<L>))==cudaSuccess,"S6A4 smem attribute");
  cudaFuncAttributes attr{};
  TORCH_CHECK(cudaFuncGetAttributes(&attr,k)==cudaSuccess,"S6A4 resources");
  // Entry (launch) registers must equal the pool the role caps tile; no local memory at all.
  TORCH_CHECK(attr.numRegs<=S6C<L>::RINIT && attr.localSizeBytes==0,"S6A4 requires the designed entry regs and no local memory");
  int occ=0;
  TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ,k,S6C<L>::NT,sizeof(S6Shared<L>))==cudaSuccess && occ>=S6C<L>::MINB,"S6A4 residency (CTAs per SM) below design");
}
void s6a4s_init() { s6_check<8>();s6_check<1>(); }
template<int L>
void s6_launch(S6Params a,cudaStream_t stream,bool pdl) {
  cudaLaunchAttribute attr{};
  attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed=1;
  cudaLaunchConfig_t c{};
  c.stream=stream;c.attrs=pdl?&attr:nullptr;c.numAttrs=pdl?1:0;
  c.gridDim=dim3(a.h.B*L);c.blockDim=dim3(128);
  TORCH_CHECK(cudaLaunchKernelEx(&c,s6a4s_pre<L>,a)==cudaSuccess,"S6A4 pre launch");
  c.gridDim=dim3(a.segments);c.blockDim=dim3(S6C<L>::NT);c.dynamicSmemBytes=sizeof(S6Shared<L>);
  TORCH_CHECK(cudaLaunchKernelEx(&c,s6a4s_segment<L>,a)==cudaSuccess,"S6A4 segment launch");
  c.gridDim=dim3(a.h.B*L);c.blockDim=dim3(128);c.dynamicSmemBytes=0;
  TORCH_CHECK(cudaLaunchKernelEx(&c,s6a4s_merge<L>,a)==cudaSuccess,"S6A4 merge launch");
}
void s6a4s_run(torch::Tensor qkv,torch::Tensor seq_lens,torch::Tensor cache_loc,
 torch::Tensor k_cache,torch::Tensor v_cache,torch::Tensor page_table,
 torch::Tensor k_scale,torch::Tensor v_scale,torch::Tensor out,
 torch::Tensor q,torch::Tensor desc,torch::Tensor bounds,torch::Tensor partial,torch::Tensor ml,
 int64_t B,int64_t L,int64_t P,double scale_log2,bool pdl) {
  TORCH_CHECK((B==16||B==32)&&(L==1||L==8)&&P>0&&(P&(P-1))==0,"S6A4 shape");
  S6Params a{};Params& p=a.h;
  p.qkv=reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  p.seq_lens=seq_lens.data_ptr<int>();p.cache_loc=cache_loc.data_ptr<int64_t>();
  p.k_cache=reinterpret_cast<uint8_t*>(k_cache.data_ptr());p.v_cache=reinterpret_cast<uint8_t*>(v_cache.data_ptr());
  p.page_table=page_table.data_ptr<int>();p.k_scale=k_scale.data_ptr<float>();p.v_scale=v_scale.data_ptr<float>();
  p.out=reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.B=B;p.Hq=16;p.Hkv=2;p.P=P;p.max_pages=page_table.size(1);p.ld_qkv=qkv.size(-1);
  p.row_bytes=128;p.lrb=7;p.off32=(k_cache.numel()<(1LL<<32)&&v_cache.numel()<(1LL<<32));
  while((1LL<<p.lp)<P) ++p.lp;
  p.scale_log2=(float)scale_log2;
  a.q=q.data_ptr<uint8_t>();a.desc=reinterpret_cast<int4*>(desc.data_ptr<int>());
  a.bounds=reinterpret_cast<int2*>(bounds.data_ptr<int>());a.partial=partial.data_ptr<float>();
  a.ml=reinterpret_cast<float2*>(ml.data_ptr<float>());a.segments=L==8?B*S6C<8>::SEG2/2:B*S6C<1>::SEG2/2;
  TORCH_CHECK(q.numel()>=B*8192 && desc.numel()>=a.segments*4 && bounds.numel()>=B*2 &&
    partial.numel()>=a.segments*L*16*64 && ml.numel()>=a.segments*L*16*2,"S6A4 scratch capacity");
  cudaStream_t stream=at::cuda::getCurrentCUDAStream();
  if(L==8) s6_launch<8>(a,stream,pdl);else s6_launch<1>(a,stream,pdl);
}
'''

CPP = r'''
void s6a4s_init();
void s6a4s_run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,
 torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,
 torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,double,bool);
'''

PYTHON = '''
S6A4_V=False
S6A4_D=False
_S6A4={"mod":None,"bank":{}}
def _s6a4s_build():
 from torch.utils.cpp_extension import load_inline
 if _S6A4["mod"] is None:
  fl=["-O3","-std=c++20","-gencode=arch=compute_90a,code=sm_90a","--expt-relaxed-constexpr","-Xptxas=-v,-warn-spills"]
  tag=hashlib.sha1((S6A4_CPP_SRC+S6A4_CUDA_SRC+" ".join(fl)).encode()).hexdigest()[:12]
  mod=load_inline(name=f"s6a4s_{tag}",cpp_sources=S6A4_CPP_SRC,cuda_sources=S6A4_CUDA_SRC,
   functions=["s6a4s_init","s6a4s_run"],extra_cflags=["-std=c++20"],extra_cuda_cflags=fl,verbose=False)
  mod.s6a4s_init()
  _S6A4["mod"]=mod
 return _S6A4["mod"]
def _s6a4s_prepare(state,t0):
 if not (S6A4_V or S6A4_D):
  return
 dev=state["router"].device
 _s6a4s_build()
 for B in (16,32):
  for L in (8,1):
   if not (S6A4_V if L==8 else S6A4_D):
    continue
   key=(str(dev),B,L)
   if key not in _S6A4["bank"]:
    G=B*(@SEG8@ if L==8 else @SEG1@)//2
    _S6A4["bank"][key]=(
     torch.empty(B*8192,dtype=torch.uint8,device=dev),
     torch.empty((G,4),dtype=torch.int32,device=dev),
     torch.empty((B,2),dtype=torch.int32,device=dev),
     torch.empty((G,L*16,64),dtype=torch.float32,device=dev),
     torch.empty((G,L*16,2),dtype=torch.float32,device=dev))
def _s6a4s_select(qkv,data,cfg,state):
 B,L=cfg["batch"],cfg["q_len"]
 if not ((L==8 and S6A4_V) or (L==1 and S6A4_D)):
  return None
 if B not in (16,32) or cfg["window"] or cfg["rope_theta"] or cfg["num_kv_heads"]!=2:
  return None
 if not _hya_wanted(qkv,data,cfg,state,True):
  return None
 key=(str(qkv.device),B,L)
 if _S6A4["mod"] is None or key not in _S6A4["bank"]:
  raise RuntimeError("S6A4 arm must be enabled before prepare: "+str(key))
 return _S6A4["bank"][key]
def _s6a4s_try(qkv,data,cfg,state):
 bank=_s6a4s_select(qkv,data,cfg,state)
 if bank is None:
  return None
 B,L=cfg["batch"],cfg["q_len"]
 out=torch.empty((B,L,cfg["num_q_heads"]*64),dtype=torch.bfloat16,device=qkv.device)
 _S6A4["mod"].s6a4s_run(qkv,data["seq_lens"],data["cache_loc"],data["k_cache"],data["v_cache"],
  data["page_table"],data["k_scale"],data["v_scale"],out,*bank,B,L,cfg["page_size"],
  float(cfg["sm_scale"]*cfg["qk_scale_factor"]*1.4426950408889634),HYA_PDL_T.get(cfg["tokens"],HYA_PDL))
 return out.view(cfg["tokens"],-1)
'''


PYTHON = PYTHON.replace("@SEG8@", str(GEOM[8]["SEG2"])).replace("@SEG1@", str(GEOM[1]["SEG2"]))
assert "@" not in PYTHON


def cuda_source(base):
    hya = literal(base, "HYA_CUDA_SRC")
    helpers = between(hya, "struct Params {", "__device__ __forceinline__ int cl_rank(")
    # This source is independent: helper text is copied now, not patched in a
    # loaded module at runtime. Attribution remains in the base notice.
    headers = between(hya, "#include <torch/extension.h>", "namespace {")
    return headers+"namespace {\nconstexpr int ROWB=128;\n"+helpers+CUDA+HOST


def marked(name, body, indent=""):
    return indent+BEGIN+name+"\n"+body+indent+END+name+"\n"


def generate(base):
    text = once(base, ' _hya_prepare(state,t0)\n', ' _hya_prepare(state,t0)\n'+
                marked("prepare", " _s6a4s_prepare(state,t0)\n", " "))
    # The old call is indented once into the fallback branch; strip_new reverses
    # that single byte as well as removing the marked additions.
    text = once(text, ' o=_hya_try(qkv,data,cfg,state,True)\n',
                marked("dispatch", ' o=_s6a4s_try(qkv,data,cfg,state)\n if o is None:\n', " ")+
                '  o=_hya_try(qkv,data,cfg,state,True)\n')
    # Reversal records exactly one indentation byte; stripping asserts it.
    block = "S6A4_CPP_SRC=r'''"+CPP+"'''\nS6A4_CUDA_SRC=r'''"+cuda_source(base)+"'''\n"+PYTHON
    text += "\n"+marked("source", block)
    ast.parse(text)
    return text


def strip_new(text):
    assert literal(text,"S6A4_V") is False and literal(text,"S6A4_D") is False
    text = re.sub(r'(?ms)^ *# S6A4 BEGIN (\w+)\n.*?^ *# S6A4 END \1\n', '', text)
    text = once(text, '  o=_hya_try(qkv,data,cfg,state,True)\n', ' o=_hya_try(qkv,data,cfg,state,True)\n')
    return text[:-1]  # the source append's one separating LF


def ship_form(lab):
    text=once(once(lab,"S6A4_V=False\n","S6A4_V=True\n"),"S6A4_D=False\n","S6A4_D=True\n")
    hya=literal(text,"HYA_CUDA_SRC")
    # The new implementation supersedes HYA's full B>1 flattened scheduler and
    # cooperative grid-wide finalize. Non-target fallback uses existing nonflat.
    fin=between(hya,"template <int L, bool WS, int NT>\n__device__ __forceinline__ void flat_finalize(",
                "template <int L, bool FLAT, int MODE = 0>\n__global__")
    hya=once(hya,fin,"")
    start=hya.index('if constexpr (FLAT && (L == 8)) {',hya.index('__global__ void __launch_bounds__(Cfg<L, FLAT>::NTX'))
    end=hya.index('\n}\n}\ntemplate <int L>\nvoid launch(',start)
    hya=hya[:start]+hya[end:]
    hya=once(hya,'const int sm_f = (int)sizeof(Smem<L, true>) + 1024, sm_n0 =', 'const int sm_n0 =')
    hya=once(hya,'p.flat ? sm_f : ((L == 8 && p.q2) ? sm_n : sm_n0)', '((L == 8 && p.q2) ? sm_n : sm_n0)')
    hya=once(hya,'occ_c = 0, occ_f = 0, nsm = 0','occ_c = 0, nsm = 0')
    hya=once(hya,'cudaFuncSetAttribute(attn_kernel<L, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_f);\n','')
    hya=once(hya,'cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_f, attn_kernel<L, true>, Cfg<L, true>::NTX, sm_f);\n','')
    hya=once(hya,'(p.flat ? occ_f : occ_c)','occ_c')
    hya=once(hya,'p.flat ? (KernFn)attn_kernel<L, true> : kn[','kn[')
    hya=hya.replace('p.flat ? Cfg<L, true>::NTX : Cfg<L, false>::NTX','Cfg<L, false>::NTX')
    text=once(text,literal(text,"HYA_CUDA_SRC"),hya)
    text=once(text,' if not window and 1<B<=512:\n  return dict(n_splits=max(1,target//pairs),flat=1,cs=0,ncl=1,nc=1)\n','')
    ast.parse(text)
    assert len(text.encode())<1_000_000, ("ship byte limit",len(text.encode()))
    assert 'attn_kernel<L, true>' not in literal(text,"HYA_CUDA_SRC")
    return text


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check",action="store_true")
    ap.add_argument("--ship",action="store_true")
    args=ap.parse_args()
    raw=BASE.read_bytes();base=raw.decode()
    assert hashlib.sha256(raw).hexdigest()==BASE_SHA256,"unreviewed base revision"
    assert "\r" not in base,"base must be LF; do not silently normalize byte proof"
    lab=generate(base)
    recovered=strip_new(lab)
    assert recovered.encode()==raw,"selector-off exact recovery failed"
    ship=ship_form(lab)
    if args.check:
        assert LAB.read_bytes()==lab.encode(),"stale lab: regenerate"
        if args.ship: assert SHIP.read_bytes()==ship.encode(),"stale ship: regenerate"
    else:
        (SHIP if args.ship else LAB).write_bytes((ship if args.ship else lab).encode())
    print(json.dumps(dict(status="PASS",base_bytes=len(raw),lab_bytes=len(lab.encode()),
      ship_bytes=len(ship.encode()),ship_net_bytes=len(ship.encode())-len(raw),
      removed_superseded_bytes=len(lab.encode())-len(ship.encode())-2,
      selector_off_exact=True,base_sha256=hashlib.sha256(raw).hexdigest())))


if __name__=="__main__":
    main()
