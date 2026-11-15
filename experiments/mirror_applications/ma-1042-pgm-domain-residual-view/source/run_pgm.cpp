#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <vector>
#include "pgm/pgm_index_dynamic.hpp"

using U=uint64_t;
using Clock=std::chrono::steady_clock;
struct Op { uint8_t kind; U key; };
struct Domain { U scale, offset; };
static constexpr size_t N=16384, NUP=128, DOM=4;
static const std::array<Domain,DOM> domains{{{3,1000000000ULL},{5,4000000000ULL},{7,9000000000ULL},{11,16000000000ULL}}};
static const U MAGIC=0x4d41313034325047ULL;

U splitmix(U &x) { U z=(x+=0x9e3779b97f4a7c15ULL); z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL; z=(z^(z>>27))*0x94d049bb133111ebULL; return z^(z>>31); }
std::vector<U> make_keys(U seed) { std::vector<U> v; v.reserve(N); U x=1000000; for(size_t i=0;i<N;i++){x+=1+(splitmix(seed)%29);v.push_back(x);} return v; }
std::vector<Op> make_updates(const std::vector<U>& k) { std::vector<Op> o; o.reserve(2*NUP); for(size_t i=0;i<NUP;i++)o.push_back({0,k.back()+1000+37*i}); for(size_t i=0;i<NUP;i++)o.push_back({1,k[100+i*113]}); return o; }
std::vector<U> final_keys(std::vector<U> k,const std::vector<Op>&ops){for(auto o:ops){if(o.kind==0)k.insert(std::lower_bound(k.begin(),k.end(),o.key),o.key);else{auto i=std::lower_bound(k.begin(),k.end(),o.key);if(i!=k.end()&&*i==o.key)k.erase(i);}}return k;}
std::optional<U> exact_pred(const std::vector<U>& k,U q){auto i=std::upper_bound(k.begin(),k.end(),q);if(i==k.begin())return std::nullopt;return *--i;}
void write_u64(std::ofstream&f,U x){f.write(reinterpret_cast<const char*>(&x),sizeof(x));}
void serialize_keys(const std::filesystem::path&p,const std::vector<U>&keys){std::filesystem::create_directories(p.parent_path());std::ofstream f(p,std::ios::binary);write_u64(f,MAGIC);write_u64(f,1);write_u64(f,keys.size());for(U x:keys)write_u64(f,x);}
void serialize_ops(const std::filesystem::path&p,const std::vector<Op>&ops,const Domain*d){std::filesystem::create_directories(p.parent_path());std::ofstream f(p,std::ios::binary);write_u64(f,MAGIC);write_u64(f,1);write_u64(f,ops.size());for(auto o:ops){f.put(char(o.kind));U x=d?d->scale*o.key+d->offset:o.key;write_u64(f,x);}}

template<size_t E> using Dyn=pgm::DynamicPGMIndex<U,U,pgm::PGMIndex<U,E,4>>;
template<class PGM> std::optional<U> pred(PGM&ix,const std::vector<U>&keys,U q){auto r=ix.search(q);auto left=(r.lo>0?r.lo-1:0);auto lo=keys.begin()+left,hi=keys.begin()+r.hi;auto i=std::lower_bound(lo,hi,q);if(i!=hi&&*i==q)return *i;if(i==lo)return std::nullopt;return *std::prev(i);}
template<class Index> void build_index(Index&idx,const std::vector<U>&keys,const std::vector<Op>&ops,const Domain*d){for(U k:keys){U x=d?d->scale*k+d->offset:k;idx.insert_or_assign(x,x);}for(auto op:ops){U x=d?d->scale*op.key+d->offset:op.key;if(op.kind==0)idx.insert_or_assign(x,x);else idx.erase(x);}}
struct Res {U shared_ram=0,independent_ram=0,shared_dynamic_ram=0,independent_dynamic_ram=0,shared_payload=0,independent_payload=0;double shared_build=0,independent_build=0,shared_update=0,independent_update=0,shared_rebuild=0,independent_rebuild=0;size_t qcount=0,correct_m=0,correct_l=0,correct_i=0,correct_no_view=0;double mp50=0,mp95=0,lp50=0,lp95=0,ip50=0,ip95=0;};
template<class V> std::pair<double,double> pct(V v){std::sort(v.begin(),v.end());return {double(v[v.size()/2])/1e6,double(v[(v.size()*95)/100])/1e6};}

template<size_t E> Res run(U seed,size_t queries,bool emit,const std::filesystem::path&outdir){
 auto base=make_keys(seed);auto ops=make_updates(base);auto final=final_keys(base,ops);
 auto b0=Clock::now();Dyn<E> shared;for(U k:base)shared.insert_or_assign(k,k);auto b1=Clock::now();
 auto u0=Clock::now();for(auto o:ops){if(o.kind==0)shared.insert_or_assign(o.key,o.key);else shared.erase(o.key);}auto u1=Clock::now();
 std::array<std::vector<U>,DOM> initd,finald;std::array<Dyn<E>,DOM> indep;double indbuild=0,indupdate=0;
 for(size_t d=0;d<DOM;d++){for(U k:base)initd[d].push_back(domains[d].scale*k+domains[d].offset);auto t0=Clock::now();for(U k:initd[d])indep[d].insert_or_assign(k,k);auto t1=Clock::now();indbuild+=std::chrono::duration<double>(t1-t0).count();auto q0=Clock::now();for(auto o:ops){U z=domains[d].scale*o.key+domains[d].offset;if(o.kind==0)indep[d].insert_or_assign(z,z);else indep[d].erase(z);}auto q1=Clock::now();indupdate+=std::chrono::duration<double>(q1-q0).count();finald[d]=final_keys(initd[d],[&](){std::vector<Op>z;for(auto o:ops)z.push_back({o.kind,domains[d].scale*o.key+domains[d].offset});return z;}());}
 using Static=pgm::PGMIndex<U,E,4>;Static shared_pgm(final.begin(),final.end());std::array<Static,DOM>ind_pgm;double shared_rebuild=0,ind_rebuild=0;auto sb0=Clock::now();shared_pgm=Static(final.begin(),final.end());auto sb1=Clock::now();shared_rebuild=std::chrono::duration<double>(sb1-sb0).count();
 for(size_t d=0;d<DOM;d++){auto t0=Clock::now();ind_pgm[d]=Static(finald[d].begin(),finald[d].end());auto t1=Clock::now();ind_rebuild+=std::chrono::duration<double>(t1-t0).count();}
 Res r;r.shared_dynamic_ram=shared.size_in_bytes();for(auto &x:indep)r.independent_dynamic_ram+=x.size_in_bytes();r.shared_ram=final.size()*sizeof(U)+shared_pgm.size_in_bytes();for(size_t d=0;d<DOM;d++)r.independent_ram+=finald[d].size()*sizeof(U)+ind_pgm[d].size_in_bytes();r.shared_build=std::chrono::duration<double>(b1-b0).count();r.shared_rebuild=shared_rebuild;r.shared_update=std::chrono::duration<double>(u1-u0).count();r.independent_build=indbuild;r.independent_rebuild=ind_rebuild;r.independent_update=indupdate;
 // Verify that the official dynamic indexes applied every update before rebuilding the predecessor predictors.
 for(U k:final)if(shared.find(k)==shared.end())throw std::runtime_error("shared dynamic update lost final key");for(auto o:ops)if(o.kind==1&&shared.find(o.key)!=shared.end())throw std::runtime_error("shared dynamic delete failed");
 for(size_t d=0;d<DOM;d++){for(U k:finald[d])if(indep[d].find(k)==indep[d].end())throw std::runtime_error("independent dynamic update lost final key");for(auto o:ops)if(o.kind==1&&indep[d].find(domains[d].scale*o.key+domains[d].offset)!=indep[d].end())throw std::runtime_error("independent dynamic delete failed");}
 if(emit){auto root=outdir/("seed_"+std::to_string(seed));auto sp=root/"shared";serialize_keys(sp/"initial_keys.bin",base);serialize_ops(sp/"updates.bin",ops,nullptr);std::filesystem::create_directories(sp);std::ofstream vf(sp/"views.bin",std::ios::binary);write_u64(vf,MAGIC);write_u64(vf,DOM);for(auto d:domains){write_u64(vf,d.scale);write_u64(vf,d.offset);}vf.close();
  std::ofstream sf(sp/"config.json");sf<<"{\"epsilon\":"<<E<<",\"epsilon_recursive\":4,\"pgm_source\":\"c6fcf3d34e55eb0061b01e2f49dfcbdb711f1407\",\"initial_keys\":"<<base.size()<<",\"inserts\":"<<NUP<<",\"deletes\":"<<NUP<<",\"rebuild\":\"DynamicPGMIndex from initial key state + ordered update stream\"}\n";sf.close();
  for(size_t d=0;d<DOM;d++){auto p=root/"independent"/("domain_"+std::to_string(d));serialize_keys(p/"initial_keys.bin",initd[d]);std::vector<Op>u;for(auto o:ops)u.push_back({o.kind,domains[d].scale*o.key+domains[d].offset});serialize_ops(p/"updates.bin",u,nullptr);std::ofstream cf(p/"config.json");cf<<"{\"epsilon\":"<<E<<",\"epsilon_recursive\":4,\"pgm_source\":\"c6fcf3d34e55eb0061b01e2f49dfcbdb711f1407\",\"initial_keys\":"<<base.size()<<",\"inserts\":"<<NUP<<",\"deletes\":"<<NUP<<"}\n";}
  r.shared_payload=0;r.independent_payload=0;for(auto const&entry:std::filesystem::recursive_directory_iterator(root)){if(entry.is_regular_file()){auto n=entry.file_size();if(entry.path().string().find("/shared/")!=std::string::npos)r.shared_payload+=n;else r.independent_payload+=n;}}
 }
 U st=seed^0x10392026;std::vector<U>xq;xq.reserve(queries);U lo=base.front(),hi=base.back()+5000;for(size_t i=0;i<queries;i++)xq.push_back(lo+splitmix(st)%(hi-lo+1));if(queries>=4){xq[0]=base.front()-1;xq[1]=base.front();xq[2]=base.front()+1;xq[3]=base.back()+1;}
 std::array<std::vector<uint64_t>,DOM> qs;for(size_t d=0;d<DOM;d++){qs[d].reserve(queries);for(U x:xq)qs[d].push_back(domains[d].scale*x+domains[d].offset);}
 std::vector<uint64_t> mt,lt,it;std::mt19937_64 rng(seed+334);std::array<int,3>arms{{0,1,2}};
 // Full exactness pass, including an intentionally wrong no-view shared index.
 for(size_t d=0;d<DOM;d++)for(size_t i=0;i<queries;i++){
  auto want=exact_pred(finald[d],qs[d][i]); auto qcanon=(qs[d][i]-domains[d].offset)/domains[d].scale; auto sm=pred(shared_pgm,final,qcanon);std::optional<U> gotm=sm?std::optional<U>(*sm*domains[d].scale+domains[d].offset):std::nullopt;
  auto sl=pred(shared_pgm,final,qcanon);std::optional<U> gotl=sl?std::optional<U>(*sl*domains[d].scale+domains[d].offset):std::nullopt;
  auto goti=pred(ind_pgm[d],finald[d],qs[d][i]);if(gotm==want)r.correct_m++;if(gotl==want)r.correct_l++;if(goti==want)r.correct_i++;if(pred(shared_pgm,final,qs[d][i])==want)r.correct_no_view++;
 }
 // Five randomized paired latency passes; transform cost is inside each measured call.
 for(int rep=0;rep<5;rep++)for(size_t d=0;d<DOM;d++)for(size_t i=0;i<queries;i++){
  std::shuffle(arms.begin(),arms.end(),rng);for(int arm:arms){auto t=Clock::now();if(arm==0){U q=(qs[d][i]-domains[d].offset)/domains[d].scale;(void)pred(shared_pgm,final,q);}else if(arm==1){U q=(qs[d][i]-domains[d].offset)/domains[d].scale;(void)pred(shared_pgm,final,q);}else (void)pred(ind_pgm[d],finald[d],qs[d][i]);auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-t).count();if(arm==0)mt.push_back(ns);else if(arm==1)lt.push_back(ns);else it.push_back(ns);}}
 auto a=pct(mt),b=pct(lt),c=pct(it);r.qcount=queries*DOM;r.mp50=a.first;r.mp95=a.second;r.lp50=b.first;r.lp95=b.second;r.ip50=c.first;r.ip95=c.second;return r;
}

template<size_t E> void output(U seed,size_t q,bool emit,const std::filesystem::path&out){auto r=run<E>(seed,q,emit,out);std::cout<<"{\"seed\":"<<seed<<",\"epsilon\":"<<E<<",\"queries_per_domain\":"<<q<<",\"shared_ram_bytes\":"<<r.shared_ram<<",\"independent_ram_bytes\":"<<r.independent_ram<<",\"shared_dynamic_ram_bytes\":"<<r.shared_dynamic_ram<<",\"independent_dynamic_ram_bytes\":"<<r.independent_dynamic_ram<<",\"shared_payload_bytes\":"<<r.shared_payload<<",\"independent_payload_bytes\":"<<r.independent_payload<<",\"shared_build_s\":"<<r.shared_build<<",\"independent_build_s\":"<<r.independent_build<<",\"shared_update_s\":"<<r.shared_update<<",\"independent_update_s\":"<<r.independent_update<<",\"shared_rebuild_s\":"<<r.shared_rebuild<<",\"independent_rebuild_s\":"<<r.independent_rebuild<<",\"queries_total\":"<<r.qcount<<",\"mirror_exact\":"<<r.correct_m<<",\"linear_exact\":"<<r.correct_l<<",\"independent_exact\":"<<r.correct_i<<",\"no_view_exact\":"<<r.correct_no_view<<",\"mirror_p50_ms\":"<<r.mp50<<",\"mirror_p95_ms\":"<<r.mp95<<",\"linear_p50_ms\":"<<r.lp50<<",\"linear_p95_ms\":"<<r.lp95<<",\"independent_p50_ms\":"<<r.ip50<<",\"independent_p95_ms\":"<<r.ip95<<"}\n";}
int main(int argc,char**argv){if(argc<4){std::cerr<<"usage: run_pgm seed query_count output_dir [epsilon] [emit]\n";return 2;}U seed=std::stoull(argv[1]);size_t q=std::stoull(argv[2]);std::filesystem::path out=argv[3];size_t e=argc>4?std::stoull(argv[4]):64;bool emit=argc>5?std::stoi(argv[5])!=0:false;switch(e){case 8:output<8>(seed,q,emit,out);break;case 16:output<16>(seed,q,emit,out);break;case 32:output<32>(seed,q,emit,out);break;case 64:output<64>(seed,q,emit,out);break;default:return 3;} }
