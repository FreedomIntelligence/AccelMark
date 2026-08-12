// views/home.js — Home page (multi-suite overview).
//
// Layout:
//   • Hero  : centered, two-line title (h1 + sub), tagline, KPI strip, CTAs.
//             Background gradient lives on body::before (no hard edges).
//   • 01    : Suite grid — uniform 3-col, 7 cards, each with a colored
//             header (letter + title + metric tag + tagline + meta line:
//             model · precision · N results · M chips) and top-6 entries
//             in the body. CTA at the bottom.
//   • 02    : Coverage by vendor — auto-fit cards, one per vendor.
//   • 05    : Contribution index — top contributors (community section).
//   • 06    : Contribute CTA — submit wizard + wanted / reproduce links.

import {
  SUITE_ORDER, SUITE_META,
  rowsForSuite, suiteFacts, chipCloudData,
  summary, recent, recentSince, formatPrimary,
  suiteChartAxisLabel, suiteChartBlurb, suiteChartHead, suiteChartPurpose,
  implOpsLabel,
} from "../data.js";
import { contributorIndex } from "../contributors.js";
import {
  esc, fmtNum, fmtDate, chipHref, buildHash,
  shortVersion, shortModel, submitterHandle,
} from "../utils.js";

const _i = (k,r) => (window._i ? window._i(k,r) : k);
const TOP_N = 8;
const RECENT_WINDOW_DAYS = 7;

// ── Distribution chart embedded in Home ──
let distChart = null, distAllSubs = [], distView = 'beeswarm';
let distNormalize = false, distMoreOpen = false;
const DIST_ADV_VIEWS = ['density', 'heatmap', 'small'];
const DIST_METRICS = {
  offline: {label:'Offline Throughput',unit:'tok/s',get:function(s){var o=s.scenarios&&s.scenarios.offline;return o&&o.is_valid?o.throughput:null;}},
  online: {label:'Online Max QPS',unit:'QPS',get:function(s){var o=s.scenarios&&s.scenarios.online;return o&&o.is_valid?o.throughput:null;}},
  sustained: {label:'Sustained Throughput',unit:'tok/s',get:function(s){var o=s.scenarios&&s.scenarios.sustained;return o&&o.is_valid?o.throughput:null;}},
  speculative: {label:'Speculative Throughput',unit:'tok/s',get:function(s){var o=s.scenarios&&s.scenarios.speculative;return o&&o.is_valid?o.throughput:null;}},
};
const DIST_VCOLS = {'NVIDIA':'#1b7a3d','Huawei':'#c2410c','AMD':'#be185d','Google':'#1d4ed8','Apple':'#78716c','Moore Threads':'#7c3aed'};
const DIST_SUITES = ['suite_A','suite_B','suite_C','suite_D','suite_E','suite_F','suite_G','suite_H'];
function distVc(v){return DIST_VCOLS[v]||'#7c3aed';}
function distFm(v){if(v===null||v===undefined)return'—';if(v>=10000)return(v/1000).toFixed(0)+'k';if(v>=1000)return(v/1000).toFixed(1)+'k';if(v>=1)return v.toFixed(1);return v.toFixed(4);}
function distHs(s){var h=5381;for(var i=0;i<s.length;i++){h=((h<<5)+h)+s.charCodeAt(i);h=h&h;}return Math.abs(h);}
function distSubOf(p){if(!p||!p.data)return null;if(p.data.submission)return p.data.submission;if(Array.isArray(p.data.value)){for(var i=p.data.value.length-1;i>=0;i--){var x=p.data.value[i];if(x&&typeof x==='object'&&x.chip)return x;}}if(Array.isArray(p.data)){for(var j=p.data.length-1;j>=0;j--){var y=p.data[j];if(y&&typeof y==='object'&&y.chip)return y;}}return null;}
function distTip(s,mt,getV,opts){if(!s)return'';var fk='color:var(--fg-faint)';var raw=getV(s);var rel='';if(opts&&opts.normalize&&opts.colMax&&opts.catField){var cat=s[opts.catField];var cm=opts.colMax[cat];if(cm){rel='<br/><span style="'+fk+'">Relative</span> <strong>'+(raw/cm*100).toFixed(1)+'%</strong> of column best';}}var suiteHead=s.suite?suiteChartHead(s.suite):'';var suitePur=s.suite?suiteChartPurpose(s.suite):'';var blurb=s.suite?suiteChartBlurb(s.suite):'';var blurbHtml=blurb?'<br/><span style="'+fk+'">'+esc(blurb)+'</span>':'';var suiteHtml=suiteHead?(suitePur?'<span style="'+fk+'">'+esc(suiteHead)+'</span><br/>'+esc(suitePur):'<span style="'+fk+'">'+esc(suiteHead)+'</span>'):'';return'<div style="font-size:11px;line-height:1.6">'+'<span style="color:'+distVc(s.chip_vendor)+'">●</span> <strong>'+esc(s.chip_vendor)+'</strong> · '+esc(s.chip)+(suiteHtml?'<br/>'+suiteHtml+blurbHtml:'')+'<br/><span style="'+fk+'">Recipe</span> <strong>'+esc(s.framework)+' '+esc(s.framework_version||'')+'</strong> · '+esc(s.precision||'-')+'<br/><span style="'+fk+'">'+esc(mt.label)+'</span> <strong style="color:var(--gold)">'+distFm(raw)+'</strong> '+esc(mt.unit)+rel+'</div>';}
function distCategoryOption(cats,catField,labelOf,valid,getV,mt,C){
  var isSuite=catField==='suite';
  var colMax={};cats.forEach(function(c){colMax[c]=0;});
  valid.forEach(function(d){var c=d[catField];var v=getV(d);if(v>0&&colMax[c]!==undefined&&v>colMax[c])colMax[c]=v;});
  function yPlot(raw,cat){return distNormalize?(colMax[cat]?raw/colMax[cat]*100:null):raw;}
  var tipOpts={normalize:distNormalize,colMax:colMax,catField:catField};
  var bestMap=new Map();valid.forEach(function(s){var k=s[catField];var vv=getV(s);if(!bestMap.has(k)||vv>bestMap.get(k).v)bestMap.set(k,{sub:s,v:vv});});
  var vendors=[...new Set(valid.map(function(d){return d.chip_vendor;}))].sort();
  var series=[];vendors.forEach(function(v){var pts=[];valid.forEach(function(d){if(d.chip_vendor!==v)return;var ci=cats.indexOf(d[catField]);if(ci<0)return;var raw=getV(d);if(!(raw>0))return;var y=yPlot(raw,d[catField]);if(y===null)return;var jt=(distHs(d.id+d.chip)%100-50)/100*0.35;var isExp=d.suite==='suite_C'&&d.precision!=='BF16'&&d.precision!=='FP16';pts.push({value:[ci+jt,y],submission:d,symbolSize:isExp?5:7,itemStyle:{opacity:isExp?.35:.65}});});if(pts.length)series.push({name:v,type:'scatter',data:pts,symbol:'circle',itemStyle:{color:distVc(v),borderColor:'rgba(0,0,0,.06)',borderWidth:1},emphasis:{scale:1.6,itemStyle:{opacity:1}},z:1});});
  vendors.forEach(function(v){var bps=[];bestMap.forEach(function(e){var s=e.sub;if(s.chip_vendor!==v)return;var ci=cats.indexOf(s[catField]);if(ci<0)return;var raw=getV(s);var y=yPlot(raw,s[catField]);if(y===null)return;var jt=(distHs(s.id+s.chip)%100-50)/100*0.35;bps.push({value:[ci+jt,y],submission:s});});if(bps.length)series.push({name:v+' best',type:'scatter',data:bps,symbol:'circle',symbolSize:13,itemStyle:{color:distVc(v),opacity:1},emphasis:{scale:1.3},z:5});});
  var yAxis=distNormalize?{type:'value',min:0,max:105,name:'% of column best',nameTextStyle:{color:C.AT,fontSize:10},axisLabel:{formatter:'{value}%',color:C.AT,fontSize:10},axisLine:{lineStyle:{color:C.AL}},splitLine:{lineStyle:{color:C.SL}}}:{type:'log',min:(function(){var all=valid.map(getV).filter(function(v){return v>0;});return all.length?Math.max(0.5,Math.min.apply(null,all)*0.5):1;})(),name:mt.label+' ('+mt.unit+')',nameTextStyle:{color:C.AT,fontSize:10},axisLabel:{formatter:function(v){return v>=1000?(v/1000).toFixed(v>=10000?0:1)+'k':v;},color:C.AT,fontSize:10},axisLine:{lineStyle:{color:C.AL}},splitLine:{lineStyle:{color:C.SL}}};
  var nCats=cats.length;
  var colTicks=cats.map(function(_,i){return i;});
  var suiteLabelFs=isSuite?10:9;
  var suiteLabelLh=isSuite?16:undefined;
  return {backgroundColor:'transparent',tooltip:{trigger:'item',backgroundColor:C.TBG,borderColor:C.TBR,textStyle:{color:C.TFG,fontSize:12},formatter:function(p){return distTip(distSubOf(p),mt,getV,tipOpts);}},legend:{data:vendors,bottom:4,textStyle:{color:C.AT,fontSize:10},itemGap:14},grid:{left:isSuite?'3%':'1.5%',right:isSuite?'3%':'1.5%',bottom:isSuite?66:72,top:'6%',containLabel:true},xAxis:{type:'value',min:-0.5,max:nCats-0.5,axisLabel:{customValues:colTicks,formatter:function(v){var i=Math.round(v);return(i>=0&&i<nCats)?labelOf(cats[i]):'';},rotate:0,fontSize:suiteLabelFs,color:C.AT,lineHeight:suiteLabelLh,align:'center',hideOverlap:false,margin:isSuite?8:8},axisTick:{show:true,alignWithLabel:true,customValues:colTicks},splitLine:{show:false},name:isSuite?'':'Framework',nameTextStyle:{color:C.AT,fontSize:10},nameGap:isSuite?6:15},yAxis:yAxis,series:series};
}
function distSyncTabUI(root){
  if(!root)return;
  var adv=DIST_ADV_VIEWS.indexOf(distView)>=0;
  if(adv)distMoreOpen=true;
  var panel=root.querySelector('#dist-more-panel');
  var toggle=root.querySelector('#dist-more-toggle');
  if(panel)panel.classList.toggle('open',distMoreOpen);
  if(toggle){toggle.classList.toggle('open',distMoreOpen);toggle.classList.toggle('has-active',adv);}
  root.querySelectorAll('.chart-tab').forEach(function(t){t.classList.toggle('active',t.dataset.distView===distView);});
  var normWrap=root.querySelector('.dist-normalize');
  if(normWrap)normWrap.hidden=distView!=='beeswarm'&&distView!=='byframework';
  var normEl=root.querySelector('#dist-normalize');
  if(normEl)normEl.checked=distNormalize;
}
function distBindClick(chart){chart.off('click');chart.on('click',function(p){var s=distSubOf(p);if(s){var h=chipHref({chip:s.chip});if(h&&h!=='#/rankings')location.hash=h;}});}
function distEnsureChart(el){if(distChart){try{if(distChart.isDisposed()){distChart=null;}else if(distChart.getDom()!==el){distChart.dispose();distChart=null;}}catch(e){distChart=null;}}if(!distChart)distChart=echarts.init(el);return distChart;}
function distShowEmpty(el,msg){if(distChart){distChart.dispose();distChart=null;}el.style.display='block';el.style.height='440px';el.style.gridTemplateColumns='';el.innerHTML='<div style="text-align:center;color:var(--fg-faint);padding:40px">'+esc(msg)+'</div>';}

function renderDistChart() {
  var el = document.getElementById('home-dist-chart'); if (!el) return;
  if (!window.DISTRIBUTION_SUBMISSIONS) { distShowEmpty(el,'Run python leaderboard/generate.py to generate distribution data'); return; }
  var fs=document.getElementById('f-suite')?.value||'',fv=document.getElementById('f-vendor')?.value||'';
  var ff=document.getElementById('f-framework')?.value||'',fp=document.getElementById('f-precision')?.value||'';
  var fc=document.getElementById('f-chip')?.value||'';
  distAllSubs = window.DISTRIBUTION_SUBMISSIONS.filter(function(s){
    if(fs&&s.suite!==fs)return false;if(fv&&s.chip_vendor!==fv)return false;
    if(ff&&s.framework!==ff)return false;if(fp&&s.precision!==fp)return false;
    if(fc&&s.chip!==fc)return false;return true;
  });
  var mtKey = document.getElementById('home-dist-metric')?.value || 'offline';
  var mt = DIST_METRICS[mtKey] || DIST_METRICS.offline;
  var getV = mt.get;
  var valid = distAllSubs.filter(function(s){return getV(s)!==null;});
  if (!valid.length) { distShowEmpty(el,'No recipes match the current filters'); return; }
  var isLight = window.matchMedia&&window.matchMedia('(prefers-color-scheme:light)').matches;
  var AL=isLight?'#e5e0d8':'#2a2d36',SL=isLight?'#ede8e0':'#20232b',AT=isLight?'#44403c':'#c8c8d0',TBG=isLight?'rgba(255,255,255,.96)':'rgba(20,22,28,.96)',TBR=isLight?'#e5e0d8':'#2a2d36',TFG=isLight?'#44403c':'#e4e4e7';
  var C={AT:AT,AL:AL,SL:SL,TBG:TBG,TBR:TBR,TFG:TFG};
  var suites = DIST_SUITES.filter(function(s){return valid.some(function(d){return d.suite===s;});});
  var vendors = [...new Set(valid.map(function(d){return d.chip_vendor;}))].sort();
  var info=document.getElementById('home-dist-info');

  if (distView === 'small') {
    if(distChart){distChart.dispose();distChart=null;}
    el.innerHTML='';el.style.display='grid';el.style.gridTemplateColumns='repeat(auto-fill,minmax(240px,1fr))';el.style.gap='6px';el.style.height='auto';
    var gxMn=valid.map(function(d){return getV(d);}).filter(function(v){return v>0;});
    gxMn=gxMn.length?Math.max(0.5,Math.min.apply(null,gxMn)*0.5):1;
    suites.forEach(function(suite){var sd=valid.filter(function(d){return d.suite===suite;});if(!sd.length)return;var inner=document.createElement('div');inner.style.cssText='background:var(--bg-elev);border:1px solid var(--border-soft);border-radius:var(--r-md);padding:6px';inner.innerHTML='<div style="font-size:.68rem;font-weight:600;color:var(--fg-strong);margin-bottom:2px;line-height:1.4">'+esc(suiteChartHead(suite))+'<br><span style="font-weight:400;color:var(--fg-muted)">'+esc(suiteChartPurpose(suite))+'</span> <span style="font-weight:400;color:var(--fg-faint)">('+sd.length+')</span></div><div style="width:100%;height:160px" class="sm-chart"></div>';el.appendChild(inner);setTimeout(function(){var el2=inner.querySelector('.sm-chart');if(!el2)return;var c=echarts.init(el2);var vends=[...new Set(sd.map(function(d){return d.chip_vendor;}))];var ser=vends.map(function(v){return{name:v,type:'scatter',data:sd.filter(function(d){return d.chip_vendor===v&&getV(d)>0;}).map(function(d){return{value:[getV(d),(d.scenarios&&d.scenarios.online&&d.scenarios.online.is_valid&&d.scenarios.online.throughput>0)?d.scenarios.online.throughput:0.05],submission:d};}),symbol:'circle',symbolSize:6,itemStyle:{color:distVc(v),opacity:.7},emphasis:{scale:1.5}};});c.setOption({tooltip:{trigger:'item',backgroundColor:TBG,borderColor:TBR,textStyle:{color:TFG,fontSize:9},formatter:function(p){return distTip(distSubOf(p),mt,getV);}},grid:{left:36,right:6,top:6,bottom:18},xAxis:{type:'log',min:gxMn,axisLabel:{fontSize:6,color:AT},splitLine:{show:false}},yAxis:{type:'log',axisLabel:{fontSize:6,color:AT},splitLine:{show:false},min:0.05},series:ser});distBindClick(c);},30);});
    if(info)info.textContent=_i('dist.info.smallchart')+' · '+valid.length+' recipes';
    var cap0=document.getElementById('home-dist-caption');
    if(cap0)cap0.textContent=_i('dist.caption.small');
    return;
  }

  /* Single-chart views: rebuild container only when coming back from grid */
  if(el.style.display==='grid'){if(distChart){distChart.dispose();distChart=null;}el.innerHTML='';}
  el.style.display='block';el.style.height=(distView==='beeswarm'||distView==='heatmap')?'460px':'440px';el.style.gridTemplateColumns='';el.style.gap='';
  var chart=distEnsureChart(el);
  var opt;

  if (distView === 'density') {
    var pts=valid.filter(function(d){return getV(d)>0;}).map(function(d){return{value:[getV(d),((distHs(d.id)%100)/100)*0.8+0.1],submission:d};});
    var allV=pts.map(function(p){return p.value[0];});
    var xMin=allV.length?Math.pow(10,Math.floor(Math.log10(Math.min.apply(null,allV)))-0.2):1;
    opt={backgroundColor:'transparent',tooltip:{trigger:'item',backgroundColor:TBG,borderColor:TBR,textStyle:{color:TFG,fontSize:12},formatter:function(p){return distTip(distSubOf(p),mt,getV);}},grid:{left:'8%',right:'5%',bottom:'8%',top:'6%',containLabel:true},xAxis:{type:'log',min:xMin,name:mt.label+' ('+mt.unit+')',nameTextStyle:{color:AT,fontSize:10},axisLabel:{formatter:function(v){return v>=1000?(v/1000).toFixed(v>=10000?0:1)+'k':v;},color:AT,fontSize:10},axisLine:{lineStyle:{color:AL}},splitLine:{lineStyle:{color:SL}}},yAxis:{show:false,min:0,max:1},series:[{type:'scatter',data:pts,symbol:'circle',symbolSize:14,itemStyle:{color:'#3b82f6',opacity:.22,borderWidth:0},emphasis:{scale:1.3,itemStyle:{opacity:.55}}}]};
  } else if (distView === 'scatter') {
    var wo=valid.filter(function(d){return d.scenarios&&d.scenarios.online&&d.scenarios.online.is_valid&&d.scenarios.online.throughput>0;});
    if(!wo.length){distShowEmpty(el,'Scatter needs recipes with valid online QPS data');return;}
    var aT=wo.map(function(d){return getV(d);}),aQ=wo.map(function(d){return d.scenarios.online.throughput;});
    var xMn=Math.max(0.5,Math.min.apply(null,aT)*0.5),yMn=Math.max(0.1,Math.min.apply(null,aQ)*0.4);
    var bestMap=new Map();wo.forEach(function(s){var k=s.suite;var vv=getV(s);if(!bestMap.has(k)||vv>bestMap.get(k).v)bestMap.set(k,{sub:s,v:vv});});
    var series=[];vendors.forEach(function(v){var pts=[],bps=[];wo.forEach(function(d){if(d.chip_vendor!==v)return;pts.push({value:[getV(d),d.scenarios.online.throughput],submission:d});});bestMap.forEach(function(e){var s=e.sub;if(s.chip_vendor!==v)return;bps.push({value:[getV(s),s.scenarios.online.throughput],submission:s});});if(pts.length)series.push({name:v,type:'scatter',data:pts,symbol:'circle',symbolSize:8,itemStyle:{color:distVc(v),opacity:.65},emphasis:{scale:1.5,itemStyle:{opacity:1}},z:1});if(bps.length)series.push({name:v+' best',type:'scatter',data:bps,symbol:'circle',symbolSize:12,itemStyle:{color:distVc(v),opacity:1},emphasis:{scale:1.3},z:5});});
    opt={backgroundColor:'transparent',tooltip:{trigger:'item',backgroundColor:TBG,borderColor:TBR,textStyle:{color:TFG,fontSize:12},formatter:function(p){return distTip(distSubOf(p),mt,getV);}},legend:{data:vendors,bottom:4,textStyle:{color:AT,fontSize:10},itemGap:14},grid:{left:'8%',right:'5%',bottom:'12%',top:'6%',containLabel:true},xAxis:{type:'log',min:xMn,name:mt.label+' ('+mt.unit+')',nameTextStyle:{color:AT,fontSize:10},axisLabel:{formatter:function(v){return v>=1000?(v/1000).toFixed(v>=10000?0:1)+'k':v;},color:AT,fontSize:10},axisLine:{lineStyle:{color:AL}},splitLine:{lineStyle:{color:SL}}},yAxis:{type:'log',min:yMn,name:'Online Max QPS',nameTextStyle:{color:AT,fontSize:10},axisLabel:{formatter:function(v){return v>=100?(v).toFixed(0):v.toFixed(1);},color:AT,fontSize:10},axisLine:{lineStyle:{color:AL}},splitLine:{lineStyle:{color:SL}}},series:series};
  } else if (distView === 'heatmap') {
    var chips=[],cm=new Map();valid.forEach(function(d){var k=d.chip;if(!cm.has(k)){cm.set(k,{chip:k,vendor:d.chip_vendor,bestThr:0});chips.push(cm.get(k));}var e=cm.get(k);if(getV(d)>e.bestThr)e.bestThr=getV(d);});
    chips.sort(function(a,b){return b.bestThr-a.bestThr;});
    var xLabs=suites.map(function(s){return suiteChartAxisLabel(s,true);}),yLabs=chips.map(function(c){return c.chip;});
    var bestLookup=new Map();valid.forEach(function(d){var k=d.chip+'\0'+d.suite;var v=getV(d);var cur=bestLookup.get(k);if(!cur||v>getV(cur))bestLookup.set(k,d);});
    var hmd=[],mx=0;chips.forEach(function(c){var row=[];suites.forEach(function(s){var best=bestLookup.get(c.chip+'\0'+s);var val=best?getV(best):null;row.push(val);if(val>mx)mx=val;});hmd.push(row);});
    var hd=[];hmd.forEach(function(row,y){row.forEach(function(v,x){if(v!==null)hd.push([x,y,v]);});});
    opt={backgroundColor:'transparent',tooltip:{trigger:'item',backgroundColor:TBG,borderColor:TBR,textStyle:{color:TFG,fontSize:12},formatter:function(p){if(!p.data||p.data[2]===null)return'';var chip=yLabs[p.data[1]],suite=suites[p.data[0]],best=bestLookup.get(chip+'\0'+suite);if(!best)return'';var blurb=suiteChartBlurb(suite);return'<strong>'+esc(best.chip)+'</strong><br/>'+esc(suiteChartHead(suite))+'<br/>'+esc(suiteChartPurpose(suite))+(blurb?'<br/><span style="color:var(--fg-faint)">'+esc(blurb)+'</span>':'')+'<br/><span style="color:var(--fg-faint)">'+esc(mt.label)+'</span> <strong>'+distFm(p.data[2])+'</strong> '+esc(mt.unit)+'<br/><span style="color:var(--fg-faint)">Recipe</span> '+esc(best.framework)+' '+esc(best.framework_version||'');}},grid:{left:180,right:28,top:10,bottom:118,containLabel:true},xAxis:{type:'category',data:xLabs,axisLabel:{fontSize:10,color:AT,fontWeight:'bold',rotate:0,lineHeight:16,interval:0,margin:12,align:'center'},position:'bottom'},yAxis:{type:'category',data:yLabs,axisLabel:{fontSize:9,color:AT,width:170,overflow:'truncate'},inverse:true},visualMap:{min:0,max:mx||1,calculable:true,orient:'vertical',right:4,bottom:'15%',textStyle:{color:AT,fontSize:9},inRange:{color:['#f0f4ff','#93c5fd','#3b82f6','#1d4ed8','#1e3a8a']}},series:[{type:'heatmap',data:hd,label:{show:true,fontSize:8,color:AT,formatter:function(p){return p.data[2]?distFm(p.data[2]):'';}},emphasis:{itemStyle:{shadowBlur:10,shadowColor:'rgba(0,0,0,.3)'}}}]};
  } else if (distView === 'byframework') {
    var fws=[...new Set(valid.map(function(d){return d.framework;}).filter(Boolean))].sort();
    opt=distCategoryOption(fws,'framework',function(f){return f;},valid,getV,mt,C);
  } else if (distView === 'beeswarm') {
    opt=distCategoryOption(suites,'suite',function(s){return suiteChartAxisLabel(s,true);},valid,getV,mt,C);
  } else {
    /* fallback */
    opt=distCategoryOption(suites,'suite',function(s){return suiteChartAxisLabel(s,true);},valid,getV,mt,C);
  }

  chart.setOption(opt,true);
  distBindClick(chart);
  chart.resize();
  var normHint=distNormalize&&(distView!=='beeswarm'&&distView!=='byframework')?' · Normalize applies to column views only':'';
  if(distNormalize&&(distView==='beeswarm'||distView==='byframework'))normHint=' · normalized to % of column best';
  if(info)info.innerHTML=_i('dist.info.showing',{n:valid.length,s:suites.length,m:esc(mt.label)})+esc(normHint);
  var cap=document.getElementById('home-dist-caption');
  if(cap){
    if(distNormalize&&(distView==='beeswarm'||distView==='byframework'))cap.textContent=_i('dist.caption.beeswarm_norm');
    else if(distView==='beeswarm'||distView==='byframework')cap.textContent=_i('dist.caption.beeswarm');
    else if(distView==='scatter')cap.textContent=_i('dist.caption.scatter');
    else if(distView==='density')cap.textContent=_i('dist.caption.density');
    else if(distView==='heatmap')cap.textContent=_i('dist.caption.heatmap');
    else cap.textContent=_i('dist.caption.small');
  }
}

function renderCommunityHub() {
  const list = contributorIndex().slice(0, 8);

  const rows = list.length ? list.map((c) => `
    <tr>
      <td class="tnum">${esc(String(c.rank))}</td>
      <td><a class="contrib-handle" href="#/contributor/${esc(c.handle)}">@${esc(c.handle)}</a></td>
      <td class="tnum">${fmtNum(c.total)}</td>
      <td class="tnum">${fmtNum(c.verified)}</td>
      <td class="tnum">${fmtNum(c.score)}</td>
    </tr>
  `).join("") : `<tr><td colspan="5" class="muted" style="padding:1rem">${_i('community.empty')}</td></tr>`;

  return `
    <section class="section community-hub">
      <div class="section-header">
        <div class="section-title">
          <span class="eyebrow">${_i('community.eyebrow')}</span>
          <h2>${_i('community.title')}</h2>
        </div>
        <a class="btn small" href="#/contributors">${_i('community.all')}</a>
      </div>
      <p class="section-sub">${_i('community.subtitle')}</p>
      <div class="contrib-table-wrap card community-hub-table">
        <table class="contrib-table community-data-table">
          <thead><tr><th>${_i('community.rank')}</th><th>${_i('community.contributor')}</th><th>${_i('community.runs')}</th><th>${_i('community.verified')}</th><th>${_i('community.score')}</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </section>
  `;
}

// Rankings-reverse hero chart — A100 vs 5090 across Suite A (bandwidth-bound),
// Suite D (compute-bound), and MSRP.  Pure inline SVG themed via CSS variables.
// Unified perspective from A100: green = A100 advantage, red = A100 disadvantage
function renderHeroChart() {
  const C = {
    blue: "var(--accent-2, #3b82f6)",
    warm: "#e8904f",
  };
  const W = 126, H = 140, PAD = { t: 20, r: 6, b: 26, l: 40 };
  const PW = W - PAD.l - PAD.r, PH = H - PAD.t - PAD.b;

  function panel(data, yFmt) {
    const max = Math.max(...data.map(d => d.v)) * 1.12;
    const y = v => PAD.t + PH - (v / max) * (PH - 4);
    const gap = 22, barW = (PW - gap) / 2;
    const xOffset = PAD.l;

    let bars = "";
    data.forEach((d, i) => {
      const bx = xOffset + i * (barW + gap);
      const by = y(d.v), bh = PAD.t + PH - by;
      bars += `<rect x="${bx.toFixed(1)}" y="${by.toFixed(1)}" width="${barW.toFixed(1)}" height="${Math.max(bh, 0).toFixed(1)}" rx="3" fill="${d.color}" opacity="0.85"/>`;
      bars += `<text x="${(bx + barW/2).toFixed(1)}" y="${(by - 6).toFixed(1)}" text-anchor="middle" font-size="10" font-weight="600" fill="var(--fg-strong)">${yFmt(d.v)}</text>`;
      bars += `<text x="${(bx + barW/2).toFixed(1)}" y="${H - 10}" text-anchor="middle" font-size="9.5" font-weight="500" fill="var(--fg-muted, #a1a1aa)">${d.label}</text>`;
    });

    let ticks = "";
    for (let i = 0; i <= 3; i++) {
      const v = (max / 3) * i;
      const ty = y(v);
      ticks += `<line x1="${PAD.l.toFixed(1)}" x2="${(W - PAD.r).toFixed(1)}" y1="${ty.toFixed(1)}" y2="${ty.toFixed(1)}" stroke="var(--border-soft, #20232b)" stroke-dasharray="2 3" opacity="0.5"/>`;
      ticks += `<text x="${(PAD.l - 5).toFixed(1)}" y="${(ty + 3).toFixed(1)}" text-anchor="end" font-size="9" fill="var(--fg-muted, #a1a1aa)">${yFmt(v)}</text>`;
    }

    return `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" style="display:block">${ticks}${bars}</svg>`;
  }

  const fmtA   = v => v >= 1000 ? (v / 1000).toFixed(1) + "k" : v.toFixed(0);
  const fmtD   = v => v.toFixed(1);
  const fmtUSD = v => v >= 1000 ? "$" + (v / 1000).toFixed(0) + "k" : "$" + v;

  const pA = panel([
    { label: "A100", v: 2701, color: C.blue },
    { label: "5090", v: 3487, color: C.warm },
  ], fmtA);
  const pD = panel([
    { label: "A100", v: 70.2, color: C.blue },
    { label: "5090", v: 54.3, color: C.warm },
  ], fmtD);
  const pP = panel([
    { label: "A100", v: 15000, color: C.blue },
    { label: "5090", v: 1999, color: C.warm },
  ], fmtUSD);

  return `<div class="hero-chart-row">
    <div class="hero-chart-panel">
      <span class="hero-chart-panel-title">Suite A · tok/s</span>${pA}
      <span class="hero-chart-panel-note disadvantage"><span class="winner-chip">A100</span> -29%</span>
    </div>
    <div class="hero-chart-panel">
      <span class="hero-chart-panel-title">Suite D · tok/s</span>${pD}
      <span class="hero-chart-panel-note advantage"><span class="winner-chip">A100</span> +29%</span>
    </div>
    <div class="hero-chart-panel">
      <span class="hero-chart-panel-title">MSRP</span>${pP}
      <span class="hero-chart-panel-note disadvantage"><span class="winner-chip">A100</span> 7.5× pricier</span>
    </div>
  </div>`;
}

export function render({ el }) {
  const s = summary();
  // 7-day momentum stat for the hero strip + standalone activity
  // ribbon below the KPIs.  The ribbon is suppressed when zero so
  // post-launch quiet weeks don't read as a regression to viewers.
  const recentCount = recentSince(RECENT_WINDOW_DAYS);
  const recentRibbon = recentCount > 0
    ? renderRecentRibbon(recentCount)
    : "";
  el.innerHTML = `
    <section class="hero">
      <div class="hero-main" style="display:flex;align-items:center;justify-content:center;gap:2rem;max-width:none;flex-wrap:nowrap">
        <div class="hero-text" style="flex:0 0 auto;min-width:0;overflow:visible">
          <h1 style="margin:0 0 0.8rem;line-height:1.35">
            ${_i('hero.title1')}
            <span class="hero-title-phrase">${_i('hero.title2')}</span>
          </h1>
          <p class="hero-sub" style="white-space:nowrap;max-width:none;line-height:1.6;margin:0.3rem 0 0">${_i('hero.subtitle')}</p>
        </div>
        <div class="hero-chart-card" style="flex-shrink:0">
          <span class="hero-chart-eyebrow">${_i('hero.chartEyebrow')}</span>
          ${renderHeroChart()}
          <p class="hero-chart-caption">${_i('hero.chartCaption')}</p>
        </div>
      </div>
      <div class="hero-stats">
        <div class="kpi"><span class="kpi-value">${fmtNum(s.total)}</span><span class="kpi-label">${_i('hero.kpi.benchmarks')}</span></div>
        <div class="kpi"><span class="kpi-value">${fmtNum(s.chips)}</span><span class="kpi-label">${_i('hero.kpi.gpus')}</span></div>
        <div class="kpi"><span class="kpi-value">${fmtNum(s.vendors)}</span><span class="kpi-label">${_i('hero.kpi.vendors')}</span></div>
        <div class="kpi"><span class="kpi-value">${fmtNum(s.suites)}</span><span class="kpi-label">${_i('hero.kpi.workloads')}</span></div>
        <div class="kpi"><span class="kpi-value">${fmtNum(s.verified)}</span><span class="kpi-label">${_i('hero.kpi.verified')}</span></div>
        ${recentCount > 0 ? `
          <div class="kpi kpi--fresh">
            <span class="kpi-value">${fmtNum(recentCount)}</span>
            <span class="kpi-label">${_i('hero.kpi.thisWeek')}</span>
          </div>
        ` : ""}
      </div>
      ${recentRibbon}
      <div class="hero-cta">
        <a class="btn primary" href="#/submit">${_i('hero.cta.submit')}</a>
        <a class="btn" href="#/compare">${_i('hero.cta.compare')}</a>
        <a class="btn" href="#/citation">${_i('hero.cta.cite')}</a>
      </div>
    </section>

    <section class="section why-submit-section">
      <div class="section-header">
        <div class="section-title">
          <span class="eyebrow">${_i('why.eyebrow')}</span>
          <h2>${_i('why.title')}</h2>
        </div>
      </div>
      <div class="why-submit-grid">
        <article class="why-submit-card">
          <h3>${_i('why.card1.title')}</h3>
          <p>${_i('why.card1.desc')}</p>
        </article>
        <article class="why-submit-card">
          <h3>${_i('why.card2.title')}</h3>
          <p>${_i('why.card2.desc')}</p>
        </article>
        <article class="why-submit-card">
          <h3>${_i('why.card3.title')}</h3>
          <p>${_i('why.card3.desc')}</p>
        </article>
      </div>
    </section>

    <section class="section">
      <div class="section-header">
        <div class="section-title">
          <span class="eyebrow">${_i('dist.eyebrow')}</span>
          <h2>${_i('dist.title')}</h2>
        </div>
        <span class="section-sub">${_i('dist.subtitle')}</span>
      </div>
      <div class="filter-row" style="margin-bottom:8px">
        <div class="fi"><label>${_i('dist.filter.suite')}</label><select id="f-suite"><option value="">${_i('dist.all')}</option></select></div>
        <div class="fi"><label>${_i('dist.filter.vendor')}</label><select id="f-vendor"><option value="">${_i('dist.all')}</option></select></div>
        <div class="fi"><label>${_i('dist.filter.framework')}</label><select id="f-framework"><option value="">${_i('dist.all')}</option></select></div>
        <div class="fi"><label>${_i('dist.filter.precision')}</label><select id="f-precision"><option value="">${_i('dist.all')}</option></select></div>
        <div class="fi"><label>${_i('dist.filter.chip')}</label><select id="f-chip"><option value="">${_i('dist.all')}</option></select></div>
        <button class="btn" id="btn-dist-reset" style="align-self:flex-end">${_i('dist.filter.reset')}</button>
      </div>
      <div class="dist-view-toolbar">
        <div class="dist-view-main">
          <span class="chart-tab active" data-dist-view="beeswarm">Beeswarm</span>
          <span class="chart-tab" data-dist-view="byframework">${_i('dist.tab.byframework')}</span>
          <span class="chart-tab" data-dist-view="scatter">Scatter</span>
          <label class="dist-normalize"><input type="checkbox" id="dist-normalize"> ${_i('dist.normalize')}</label>
          <span class="info-btn">ⓘ<span class="info-pop">${_i('dist.info.all')}</span></span>
        </div>
        <span class="dist-metric-bar"><span class="dist-metric-label">${_i('dist.metric.label')}</span><select id="home-dist-metric"><option value="offline">${_i('dist.metric.offline')}</option><option value="online">${_i('dist.metric.online')}</option><option value="sustained">${_i('dist.metric.sustained')}</option><option value="speculative">${_i('dist.metric.speculative')}</option></select></span>
      </div>
      <div class="dist-view-more-bar">
        <span class="dist-view-more-toggle" id="dist-more-toggle" role="button" tabindex="0"><span class="chev">▸</span> ${_i('dist.moreviews')}</span>
      </div>
      <div class="dist-view-more-panel" id="dist-more-panel">
        <span class="chart-tab" data-dist-view="density">Density</span>
        <span class="chart-tab" data-dist-view="heatmap">Heatmap</span>
        <span class="chart-tab" data-dist-view="small">By Suite</span>
      </div>
      <div id="home-dist-chart" style="width:100%;height:440px"></div>
      <div id="home-dist-info"></div>
      <p id="home-dist-caption" class="figure-caption">${_i('home.figure1')} (framework × precision × hardware); larger markers denote best-in-class per vendor. Y-axis log-scaled.</p>
    </section>

    <section class="section">
      <div class="section-header">
        <div class="section-title">
          <span class="eyebrow">${_i('workloads.eyebrow')}</span>
          <h2>${_i('workloads.title')}</h2>
        </div>
        <span class="section-sub">${_i('workloads.subtitle')}</span>
      </div>
      <div class="suite-grid" id="suite-grid"></div>
    </section>

    <section class="section">
      <div class="section-header">
        <div class="section-title">
          <span class="eyebrow">${_i('coverage.eyebrow')}</span>
          <h2>${_i('coverage.title')}</h2>
        </div>
        <span class="section-sub">${_i('coverage.subtitle')}</span>
      </div>
      <div class="chip-cloud" id="chip-cloud"></div>
      <div class="cloud-legend" id="cloud-legend"></div>
    </section>

    <section class="section">
      <div class="section-header">
        <div class="section-title">
          <span class="eyebrow">${_i('recent.eyebrow')}</span>
          <h2>${_i('recent.title')}</h2>
        </div>
        <a class="btn small" href="#/rankings">${_i('recent.seeall')}</a>
      </div>
      <div class="recent-list" id="recent-list"></div>
    </section>

    ${renderCommunityHub()}

    <section class="section submit-section">
      <div class="submit-card">
        <span class="eyebrow">${_i('contribute.eyebrow')}</span>
        <h2 class="submit-title">${_i('contribute.title')}</h2>
        <p class="submit-body">${_i('contribute.body')}</p>
        <ol class="submit-quickstart">
          <li>${_i('contribute.step1')}</li>
          <li>${_i('contribute.step2')}</li>
          <li>${_i('contribute.step3')}</li>
        </ol>
        <p class="submit-body submit-body-foot muted">
          ${_i('contribute.foot')}
          <a href="https://github.com/FreedomIntelligence/AccelMark/blob/main/CONTRIBUTING.md" target="_blank" rel="noopener">${_i('contribute.guide')}</a>
        </p>
        <div class="submit-cta">
          <a class="btn primary" href="#/submit">${_i('contribute.cta1')}</a>
          <a class="btn" href="#/wanted">${_i('contribute.cta2')}</a>
          <a class="btn" href="#/reproduce">${_i('contribute.cta3')}</a>
        </div>
      </div>
    </section>
  `;

  const grid = el.querySelector("#suite-grid");
  for (const suiteId of SUITE_ORDER) {
    grid.appendChild(renderSuiteCard(suiteId));
  }

  const cloud = el.querySelector("#chip-cloud");
  const legend = el.querySelector("#cloud-legend");
  renderChipCloud(cloud, legend);

  const recentEl = el.querySelector("#recent-list");
  for (const row of recent(8)) {
    recentEl.appendChild(renderRecentRow(row));
  }

  // Populate filter dropdowns
  if (window.DISTRIBUTION_SUBMISSIONS) {
    var subs = window.DISTRIBUTION_SUBMISSIONS;
    var uniq = function(f){return [...new Set(subs.map(function(x){return x[f];}).filter(Boolean))].sort();};
    ['f-suite','f-vendor','f-framework','f-precision','f-chip'].forEach(function(id){
      var field = id === 'f-suite' ? 'suite' : id === 'f-vendor' ? 'chip_vendor' : id === 'f-framework' ? 'framework' : id === 'f-precision' ? 'precision' : 'chip';
      var sel = el.querySelector('#'+id); if (!sel) return;
      sel.innerHTML = '<option value="">'+_i('dist.all')+'</option>' + uniq(field).map(function(v){return '<option value="'+esc(v)+'">'+esc(v)+'</option>';}).join('');
    });
  }

  // Render distribution chart
  if (!el.__distBound) {
    el.__distBound = true;
    el.addEventListener('change', function(e) {
      if (e.target && e.target.id === 'dist-normalize') { distNormalize = e.target.checked; renderDistChart(); return; }
      if (e.target && (e.target.id === 'home-dist-metric' || e.target.id && e.target.id.startsWith('f-'))) renderDistChart();
    });
    el.addEventListener('click', function(e) {
      if (e.target.closest('#dist-more-toggle')) {
        distMoreOpen = !distMoreOpen;
        distSyncTabUI(el);
        return;
      }
      var tab = e.target.closest('.chart-tab');
      if (tab && tab.dataset.distView) {
        distView = tab.dataset.distView;
        if (DIST_ADV_VIEWS.indexOf(distView) < 0) distMoreOpen = false;
        distSyncTabUI(el);
        renderDistChart();
        return;
      }
      if (e.target.closest('#btn-dist-reset')) { el.querySelectorAll('.filter-row select').forEach(function(s){s.value='';}); renderDistChart(); }
    });
    el.addEventListener('keydown', function(e) {
      if (e.target && e.target.id === 'dist-more-toggle' && (e.key === 'Enter' || e.key === ' ')) {
        e.preventDefault();
        distMoreOpen = !distMoreOpen;
        distSyncTabUI(el);
      }
    });
    window.addEventListener('resize', function() { if (distChart) distChart.resize(); });
    window.matchMedia('(prefers-color-scheme:light)').addEventListener('change', function() { renderDistChart(); });
  }
  setTimeout(function(){
    distSyncTabUI(el);
    renderDistChart();
  }, 100);
}

// Hero ribbon directly below the KPI strip.  Tells contributors that
// fresh submissions land on the front page (not buried in a per-suite
// view) and gives a one-click way to scan the latest activity sorted
// by date across the canonical default suite.
function renderRecentRibbon(n) {
  const label = n === 1 ? "submission" : "submissions";
  const href = buildHash("/rankings", { suite: SUITE_ORDER[0], sort: "date:desc" });
  return `
    <a class="hero-recent-ribbon"
       href="${esc(href)}"
       title="Open results sorted by submission date.">
      <span class="hero-recent-dot" aria-hidden="true"></span>
      <span class="hero-recent-text">
        <strong>${fmtNum(n)}</strong> new ${label}
        <span class="hero-recent-window">in the last ${RECENT_WINDOW_DAYS} days</span>
      </span>
      <span class="hero-recent-cta" aria-hidden="true">See latest →</span>
    </a>
  `;
}

function renderSuiteCard(suiteId) {
  const meta = SUITE_META[suiteId];
  const facts = suiteFacts(suiteId);
  const top = rowsForSuite(suiteId).slice(0, TOP_N);
  const empty = top.length === 0;

  const card = document.createElement("article");
  card.className = "card suite-card" + (empty ? " empty" : "");
  card.setAttribute("data-suite", meta.letter);

  const rankingsHref = buildHash("/rankings", { suite: suiteId });

  const metaLine = renderSuiteMeta(suiteId, facts);
  const header = `
    <div class="suite-card-head">
      <div class="suite-head-row1">
        <div class="suite-head-left">
          <span class="suite-letter">${esc(meta.letter)}</span>
          <span class="suite-title">${esc(_i('suite.' + suiteId + '.title'))}</span>
        </div>
        <span class="suite-metric-tag">${esc(meta.primary.label)}</span>
      </div>
      <p class="suite-head-tagline">${esc(_i('suite.' + suiteId + '.tagline'))}</p>
      <div class="suite-head-meta">${metaLine}</div>
    </div>
  `;

  if (empty) {
    card.innerHTML = `
      ${header}
      <div class="suite-card-body">${_i('workloads.awaiting')}</div>
      <div class="suite-card-foot"><a class="cta" href="${rankingsHref}">${_i('workloads.viewfull')}</a></div>
    `;
    return card;
  }

  const body = top.map((r, i) => renderLbRow(r, suiteId, i + 1)).join("");
  card.innerHTML = `
    ${header}
    <div class="suite-card-body">${body}</div>
    <div class="suite-card-foot"><a class="cta" href="${rankingsHref}">${_i('workloads.viewfull')}</a></div>
  `;
  return card;
}

function renderSuiteMeta(suiteId, facts) {
  const wl = SUITE_META[suiteId] && SUITE_META[suiteId].workload;
  const items = [];
  if (facts.model) {
    items.push(`<span class="meta-item"><strong>${esc(shortModel(facts.model))}</strong></span>`);
  }
  if (facts.precision) {
    items.push(`<span class="meta-item">${esc(facts.precision)} baseline</span>`);
  }
  if (wl && wl.inputTokens && wl.outputTokens) {
    items.push(`<span class="meta-item">${esc(wl.inputTokens)} → ${esc(wl.outputTokens)} tok</span>`);
  }
  if (facts.submissions) {
    items.push(`<span class="meta-item"><strong>${fmtNum(facts.submissions)}</strong> ${_i('home.resultsLabel')}</span>`);
  }
  if (facts.chips) {
    items.push(`<span class="meta-item"><strong>${fmtNum(facts.chips)}</strong> ${_i('home.chipsLabel')}</span>`);
  }
  return items.join("");
}

// Single ranked row used by suite cards.  Plain click → run modal.
// Anchor href stays pointed at the chip detail page so Cmd-click still
// opens the chip-level overview in a new tab.
function renderLbRow(row, suiteId, rank) {
  const meta = SUITE_META[suiteId];
  const value = row[meta.primary.key];
  const display = formatPrimary(value, suiteId);
  const { num, unit } = splitNumUnit(display);
  const medal = rank === 1 ? "gold" : rank === 2 ? "silver" : rank === 3 ? "bronze" : "";
  const featured = rank === 1 ? " lb-row--featured" : "";
  const runId = row.run_id || row.submission || "";
  const ver = shortVersion(row.framework_version);
  const fw = row.framework || "";
  const ops = implOpsLabel(row);
  const chipRecipe = fw
    ? `${esc(row._chip_label)} · <span class="lb-row-fw"><span style="white-space:nowrap">${esc(fw)}${ver ? ` <span class="fw-ver">${esc(ver)}</span>` : ""}</span>${ops ? ` <span class="fw-ops">${esc(ops)}</span>` : ""}</span>`
    : esc(row._chip_label);
  const a11yLabel = `Open run details for ${row._chip_label}${fw ? " on " + fw : ""}`;
  return `
    <div class="lb-row${featured}"
         role="button"
         tabindex="0"
         aria-label="${esc(a11yLabel)}"
         data-open-run="${esc(runId)}">
      <span class="lb-row-rank ${medal}">${rank}</span>
      <span class="lb-row-main">
        <a class="lb-row-name" href="${chipHref(row)}">
          <span class="lb-row-chip">${chipRecipe}</span>
        </a>
        ${renderRecipeSub(row)}
      </span>
      <span class="lb-row-score">
        <span class="score-val">${esc(num)}</span>
        ${unit ? `<span class="score-unit">${esc(unit)}</span>` : ""}
      </span>
    </div>
  `;
}

// Sub block under recipe line: vendor · precision · @submitter
function renderRecipeSub(row) {
  const precision = row.precision
    ? `<span class="sub-sep">·</span><span class="prec-tag">${esc(row.precision)}</span>`
    : "";
  const handle = submitterHandle(row.submitted_by);
  const byline = handle
    ? `<span class="lb-row-byline">@${esc(handle)}</span>`
    : "";
  return `
    <span class="lb-row-sub">
      <span class="vendor-dot" data-vendor="${esc(row.vendor)}"></span>
      <span class="vendor-name">${esc(row.vendor)}</span>${precision}
    </span>
    ${byline}
  `;
}

function renderChipCloud(container, legendEl) {
  const chips = chipCloudData();
  for (const c of chips) {
    const a = document.createElement("a");
    a.className = `chip-tile size-${c.size}`;
    // Each tile is a navigational link to the chip's overview page.
    // From there users can drill into any specific run or jump to
    // Compare with this chip pre-selected, but the home landing stays
    // a "browse chips" experience rather than a "start comparing"
    // funnel — matches user expectations for a clickable chip name.
    a.href = `#/chip/${c.slug}`;
    a.setAttribute("data-vendor", c.vendor);
    const subL = c.submissions === 1 ? "submission" : "submissions";
    const suiteL = c.suites.length === 1 ? "suite" : "suites";
    const variantPart = c.variants > 1 ? ` · ${c.variants} chip-count variants` : "";
    a.setAttribute("title",
      `${c.label}: ${c.submissions} ${subL} across ` +
      `${c.suites.length} ${suiteL}${variantPart}`);
    a.innerHTML = `
      <span class="chip-tile-name">${esc(c.label)}</span>
      <span class="chip-tile-count">${fmtNum(c.submissions)}</span>
    `;
    container.appendChild(a);
  }

  // Vendor legend below the cloud
  if (!legendEl) return;
  const byVendor = new Map();
  for (const c of chips) {
    const v = byVendor.get(c.vendor) || { vendor: c.vendor, chips: 0, submissions: 0 };
    v.chips += 1;
    v.submissions += c.submissions;
    byVendor.set(c.vendor, v);
  }
  const vendors = Array.from(byVendor.values()).sort((a, b) => b.submissions - a.submissions);
  legendEl.innerHTML = vendors.map((v) => {
    const chipsLbl = v.chips === 1 ? _i('home.chipLabel') : _i('home.chipsLabel');
    const subsLbl  = v.submissions === 1 ? _i('home.resultLabel') : _i('home.resultsLabel');
    return `
      <span class="cloud-legend-item" data-vendor="${esc(v.vendor)}">
        <span class="dot"></span>
        <span class="name">${esc(v.vendor)}</span>
        <span class="meta">${fmtNum(v.chips)} ${chipsLbl} · ${fmtNum(v.submissions)} ${subsLbl}</span>
      </span>
    `;
  }).join("");
}

function renderRecentRow(row) {
  const meta = SUITE_META[row.suite];
  const metricVal = meta ? row[meta.primary.key] : row.primary_metric;
  const display = formatPrimary(metricVal, row.suite);
  const { num, unit } = splitNumUnit(display);
  const suiteLabel = row.suite.replace("suite_", "Suite ");
  const letter = meta ? meta.letter : "·";
  const handle = submitterHandle(row.submitted_by);
  const ver = shortVersion(row.framework_version);
  const fw = row.framework || "";
  const ops = implOpsLabel(row);
  const chipRecipe = fw
    ? `${esc(row._chip_label)} · <span class="lb-row-fw"><span style="white-space:nowrap">${esc(fw)}${ver ? ` <span class="fw-ver">${esc(ver)}</span>` : ""}</span>${ops ? ` <span class="fw-ops">${esc(ops)}</span>` : ""}</span>`
    : esc(row._chip_label);
  // Mirrors renderLbRow: outer <div> = run-modal trigger, inner chip-name
  // <a> escapes via modal.js nested-anchor rule to navigate to /chip/<slug>.
  const wrap = document.createElement("div");
  wrap.className = "lb-row";
  wrap.setAttribute("data-suite", letter);
  wrap.setAttribute("data-open-run", row.run_id || row.submission || "");
  // a11y: same role + keyboard hooks as renderLbRow so the recent
  // activity list is fully tab-able and screen-reader-discoverable.
  wrap.setAttribute("role", "button");
  wrap.setAttribute("tabindex", "0");
  wrap.setAttribute("aria-label", `Open run details for ${row._chip_label} on ${suiteLabel}`);
  const bylineBits = [];
  if (handle) bylineBits.push(`@${esc(handle)}`);
  bylineBits.push(esc(fmtDate(row.date)));
  const byline = `<span class="lb-row-byline">${bylineBits.join(" · ")}</span>`;
  const precision = row.precision
    ? `<span class="sub-sep">·</span><span class="prec-tag">${esc(row.precision)}</span>`
    : "";
  wrap.innerHTML = `
    <span class="lb-row-rank suite-tag-rank" aria-hidden="true">${esc(letter)}</span>
    <span class="lb-row-main">
      <a class="lb-row-name" href="${chipHref(row)}" title="${esc(row._chip_label)}${fw ? " · " + fw + (ver ? " " + ver : "") : ""}">
        <span class="lb-row-chip">${chipRecipe}</span>
      </a>
      <span class="lb-row-sub">
        <span class="vendor-dot" data-vendor="${esc(row.vendor)}"></span>
        <span class="vendor-name">${esc(row.vendor)}</span>${precision}
        <span class="sub-sep">·</span>
        <span class="suite-tag">${esc(suiteLabel)}</span>
      </span>
      ${byline}
    </span>
    <span class="lb-row-score">
      <span class="score-val">${esc(num)}</span>
      ${unit ? `<span class="score-unit">${esc(unit)}</span>` : ""}
    </span>
  `;
  return wrap;
}

// "5,731 tok/s" → { num: "5,731", unit: "tok/s" }
function splitNumUnit(s) {
  if (!s) return { num: "-", unit: "" };
  const idx = s.search(/\s[A-Za-z%]/);
  if (idx === -1) return { num: s, unit: "" };
  return { num: s.slice(0, idx), unit: s.slice(idx + 1) };
}
