/* Offline interactions: educational simulation and read-only evidence import. */
(() => {
  'use strict';
  const $=id=>document.getElementById(id), P=window.PROJECT, M=window.PhyMath, G=window.GLOSSARY;
  const num=(v,d=2)=>Number(v).toLocaleString('id-ID',{maximumFractionDigits:d});
  const put=(id,text)=>{if($(id)) $(id).textContent=text;};
  const node=(tag,text,cls)=>{const el=document.createElement(tag);if(text!==undefined)el.textContent=text;if(cls)el.className=cls;return el;};
  const config=P.config, profile=P.profile.model;
  document.querySelectorAll('[data-snapshot-date]').forEach(el=>el.textContent=P.snapshot_date);
  put('software-progress',`${P.status.software_release.completion_percent}%`);
  put('research-progress',`${P.status.overall_progress_percent}%`);
  if($('software-bar'))$('software-bar').style.width=`${P.status.software_release.completion_percent}%`;
  if($('research-bar'))$('research-bar').style.width=`${P.status.overall_progress_percent}%`;
  put('status-note',`Rilis lokal: ${P.status.software_release.completion_percent}%; riset keseluruhan: ${P.status.overall_progress_percent}%; kesiapan kompetisi: ${P.status.competition_readiness_percent}%. Training Kaggle, OOF lengkap, dan leaderboard belum tersedia.`);
  put('parameter-count',`${num(profile.parameters/1e6,2)} M`);
  put('gmacs',`${num(profile.forward_macs_batch/1e9,2)}`);
  put('weights-memory',`${num(profile.weight_mib_fp32)} MiB`);
  put('state-memory',`${num(profile.adamw_plus_ema_state_mib_fp32)} MiB`);
  put('local-hardware',`${P.profile.hardware.torch} · ${P.profile.hardware.cuda_available?'CUDA tersedia':'CPU saja; CUDA tidak tersedia'}`);
  put('paper-parameters',num(profile.parameters,0));
  put('paper-macs',`${num(profile.forward_macs_batch/1e9,3)} GMAC / ≈${num(profile.forward_flops_approx_batch/1e9,3)} GFLOP`);
  put('paper-weights',`${num(profile.weight_mib_fp32)} MiB`);
  put('paper-states',`${num(profile.adamw_plus_ema_state_mib_fp32)} MiB`);
  put('paper-input',`${num(profile.input_mib_fp32)} MiB`);

  const tooltip=$('term-tip'); let tipTrigger=null;
  function hideTip(){tooltip.hidden=true;if(tipTrigger)tipTrigger.removeAttribute('aria-describedby');tipTrigger=null;}
  function showTip(el){
    const key=el.dataset.term; if(!G[key])return;
    tooltip.textContent=`${key} — ${G[key]}`;tooltip.hidden=false;tipTrigger=el;el.setAttribute('aria-describedby','term-tip');
    const r=el.getBoundingClientRect(),tr=tooltip.getBoundingClientRect();
    tooltip.style.left=`${Math.max(8,Math.min(innerWidth-tr.width-8,r.left))}px`;
    tooltip.style.top=`${r.bottom+tr.height+10>innerHeight?Math.max(8,r.top-tr.height-8):r.bottom+8}px`;
  }
  function prepareTerms(scope=document){
    scope.querySelectorAll('[data-term]:not(button)').forEach(el=>{const b=node('button',el.textContent,'term');b.type='button';b.dataset.term=el.dataset.term;el.replaceWith(b);});
    scope.querySelectorAll('button[data-term]:not([data-ready])').forEach(el=>{el.dataset.ready='1';el.classList.add('term');el.addEventListener('mouseenter',()=>showTip(el));el.addEventListener('mouseleave',hideTip);el.addEventListener('focus',()=>showTip(el));el.addEventListener('blur',hideTip);el.addEventListener('click',()=>showTip(el));});
  }
  document.addEventListener('keydown',e=>{if(e.key==='Escape')hideTip();});
  document.addEventListener('click',e=>{if(!e.target.closest('[data-term]'))hideTip();});
  window.addEventListener('scroll',hideTip,{passive:true});
  prepareTerms();
  // Annotate further technical terms in prose, without touching code/controls.
  const keys=Object.keys(G).sort((a,b)=>b.length-a.length), escaped=keys.map(k=>k.replace(/[.*+?^${}()|[\]\\]/g,'\\$&'));
  const pattern=new RegExp(`(?<![\\w-])(${escaped.join('|')})(?![\\w-])`,'gi');
  const keyMap=Object.fromEntries(keys.map(k=>[k.toLowerCase(),k]));
  function annotateTerms(scope){
  const walker=document.createTreeWalker(scope,NodeFilter.SHOW_TEXT);
  const texts=[];while(walker.nextNode())texts.push(walker.currentNode);
  texts.forEach(text=>{
    if(!text.parentElement || text.parentElement.closest('button,a,code,pre,svg,script,input,select,textarea,label,[data-term],#glossary,.equation'))return;
    const str=text.nodeValue;pattern.lastIndex=0;let match,last=0,frag=document.createDocumentFragment(),found=false;
    while((match=pattern.exec(str))){found=true;frag.append(document.createTextNode(str.slice(last,match.index)));const b=node('button',match[0],'term');b.type='button';b.dataset.term=keyMap[match[0].toLowerCase()];frag.append(b);last=pattern.lastIndex;}
    if(found){frag.append(document.createTextNode(str.slice(last)));text.replaceWith(frag);}
  });prepareTerms(scope);
  }
  annotateTerms(document.querySelector('main'));

  document.querySelectorAll('[data-print]').forEach(b=>b.addEventListener('click',()=>window.print()));
  const steps=[
    ['Foto + metadata','RGB penuh dari 2–5 pandangan; sample_id dan PPM kamera. Label laboratorium hanya untuk training.','Katalog foto → sampel → kamera → PPM; target 11 titik per tanah.','Mencocokkan ID dan memisahkan label train dari foto test mencegah salah mapping. Audit gagal jika metadata penting tidak valid.','src/phygrainnet/data/catalog.py'],
    ['Kalibrasi fisik','Foto native dan PPM yang telah disesuaikan dengan ukuran file aktual.','Crop border 6% → resample 10 px/mm → gray-world → image canonical di cache.','Tanah berukuran sama harus mempunyai ukuran piksel konsisten meski kamera berubah. Cache bergantung identitas foto serta parameter fisik.','src/phygrainnet/data/imaging.py'],
    ['Tile dua skala','Image canonical seluruh foto satu sampel.','Training: B×T×S×3×256×256; bidang 25,6 mm dan 102,4 mm.','Detail lokal dan konteks dipusatkan pada lokasi yang sama. Sampling dari semua foto menambah keragaman tanpa menambah jumlah label.','src/phygrainnet/training/data.py'],
    ['Encoder tekstur','Tile RGB; bentuk B×T×S×3×256×256.','Sobel magnitude + Laplacian → 5 kanal → GrainBlock → mean/std spasial → 448 fitur per skala.','Filter analitik tidak memakai bobot pretrained. Dua skala berbagi encoder, tetapi tetap dihitung keduanya.','src/phygrainnet/models/phygrainnet.py'],
    ['Satukan pandangan','448 fitur per skala → concat 896 → proyeksi embedding tile 256.','Pool mean / std / attention → 768 → MLP embedding sampel 256.','Himpunan tile menjadi satu representasi sampel. Inference mengodekan tile per chunk, lalu mempool seluruh tile bersama.','src/phygrainnet/models/phygrainnet.py'],
    ['Kurva GSD + CSV','Embedding sampel 256; head linear menghasilkan 11 logit.','Softmax → 11 massa total 100 → cumsum → CDF; validasi ID, schema, batas dan endpoint sebelum CSV.','Satu sampel satu kurva. Kolom diameter harus dalam urutan template; 200 mm tepat 100. Nilai non-finite ditolak.','src/phygrainnet/submission.py']
  ];
  function selectStep(i){
    const [title,input,output,reason,file]=steps[i], container=$('flow-detail');if(!container)return;
    container.replaceChildren();container.append(node('h3',`${i+1}. ${title}`));
    const row=node('div',undefined,'flow-detail');[[ 'Masukan',input],['Keluaran',output]].forEach(([name,text])=>{const io=node('div',undefined,'io');io.append(node('strong',name),node('p',text));row.append(io);});
    container.append(row,node('p',reason,'small'),node('code',file));
    document.querySelectorAll('[data-step]').forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.step===i)));
    annotateTerms(container);
  }
  document.querySelectorAll('[data-step]').forEach(b=>b.addEventListener('click',()=>selectStep(+b.dataset.step)));selectStep(0);
  if($('shape-track')){
    const b=config.training.batch_size,t=config.training.tiles_per_sample,s=config.data.fields_mm.length;
    [['Input',1,`${b}×${t}×${s}×3×256×256`],['Texture',.94,`${b*t} per skala ×5×256×256`],['Encoder',.8,`${b*t}×224×8×8`],['Spatial pool',.6,`${b*t}×448 per skala`],['Tile project',.43,`${b}×${t}×256`],['Bag pool',.3,`${b}×768`],['MLP',.22,`${b}×256`],['CDF',.08,`${b}×11`]].forEach(([label,width,shape])=>{const row=node('div',undefined,'shape-row'),bar=node('span');bar.style.width=`${width*100}%`;row.append(node('strong',label),bar,node('code',shape));$('shape-track').append(row);});
  }
  if($('ablation-rows'))P.ablations.forEach(a=>{const tr=node('tr');tr.append(node('td',a.id),node('td',a.description));$('ablation-rows').append(tr);});
  if($('gate-grid'))Object.entries(P.status.gates).forEach(([key,g])=>{const a=node('article',undefined,'panel');a.append(node('span',g.status,'badge pending'),node('h3',`${key} · ${g.name}`),node('p',`${num(g.completion*100,0)}% item evidence. Kriteria lengkap di docs/READINESS_GATE.md.`, 'small muted'));$('gate-grid').append(a);});
  if($('hyperparameter-rows')){
    const t=config.training,d=config.data,inf=config.inference,m=config.model;
    [['Seed',P.status.default_seed,'Seed awal; finalis 42/123/2026'],['Skala canonical',`${d.target_ppm} px/mm`,'0,1 mm/piksel'],['Bidang / tile',`${d.fields_mm.join(' / ')} mm; ${d.tile_px} px`,'Dua crop konsentris'],['Batch / tile / skala',`${t.batch_size} / ${t.tiles_per_sample} / ${d.fields_mm.length}`,'Satu item adalah tanah'],['Epoch / step',`${t.epochs} / ${t.steps_per_epoch}`,'Sampling acak, 2.000 update/fold'],['Optimizer','AdamW','Learning rate adaptif'],['Learning rate',t.learning_rate,'Warm-up + cosine'],['Weight decay',t.weight_decay,'Regularisasi'],['Warm-up',`${t.warmup_steps} step`,'10% update awal/fold'],['Clip norm',t.gradient_clip_norm,'Setelah unscale AMP'],['EMA',t.ema_decay,'Bobot rata-rata bergerak'],['Dropout',m.dropout,'Sebelum/sesudah MLP'],['Tile mixing / jitter',`${t.mix_prob} / ±${t.scale_jitter*100}%`,'Augmentasi'],['Eval interval',`${t.eval_every} epoch`,'Hanya monitoring, fixed schedule'],['TTA / grid limit / chunk',`${inf.tta} / ${inf.max_tiles_per_image} / ${inf.chunk}`,'Rotasi / per foto / tile encoder'],['CV',`${config.cv.strategy} × ${config.cv.n_folds}`,'Kelompok ID utuh'],['AMP / full fit',`${t.amp} / ${t.full_fit}`,'CUDA FP16 / tanpa full fit tambahan']].forEach(values=>{const tr=node('tr');values.forEach(v=>tr.append(node('td',String(v))));$('hyperparameter-rows').append(tr);});
  }

  const NS='http://www.w3.org/2000/svg';
  function svgNode(tag,attrs,text){const e=document.createElementNS(NS,tag);Object.entries(attrs||{}).forEach(([k,v])=>e.setAttribute(k,String(v)));if(text!==undefined)e.textContent=text;return e;}
  function setupSvg(svg,height=285){const width=Math.max(280,Math.round(svg.getBoundingClientRect().width));svg.setAttribute('viewBox',`0 0 ${width} ${height}`);svg.replaceChildren();return {width,height};}
  function cdfChart(svg,target,pred){
    if(!svg)return;const {width:w,height:h}=setupSvg(svg),l=46,r=18,top=18,bottom=h-48;
    const x=v=>l+(Math.log10(v)-Math.log10(M.diameters[0]))/5*(w-l-r),y=v=>bottom-v/100*(bottom-top);
    [0,25,50,75,100].forEach(v=>{svg.append(svgNode('line',{x1:l,y1:y(v),x2:w-r,y2:y(v),class:'axis'}),svgNode('text',{x:l-8,y:y(v)+4,'text-anchor':'end'},v));});
    const ticks=w<430?[.002,.063,2,200]:[.002,.02,.2,2,20,200];
    ticks.forEach((d,i)=>svg.append(svgNode('text',{x:x(d),y:bottom+21,'text-anchor':i===0?'start':i===ticks.length-1?'end':'middle'},num(d,4))));
    svg.append(svgNode('text',{x:l,y:12},'Lolos (% massa)'),svgNode('text',{x:(w+l-r)/2,y:h-6,'text-anchor':'middle'},'Diameter (mm) · sumbu log₁₀'));
    [[target,'target'],[pred,'prediction']].forEach(([c,cls])=>{svg.append(svgNode('path',{d:c.map((v,i)=>`${i?'L':'M'}${x(M.diameters[i])},${y(v)}`).join(' '),class:cls}));c.forEach((v,i)=>{const dot=svgNode('circle',{cx:x(M.diameters[i]),cy:y(v),r:3,fill:cls==='target'?'#12654f':'#ad581c'});dot.append(svgNode('title',{},`${cls==='target'?'Referensi':'Prediksi'}: ${M.diameters[i]} mm = ${num(v)}%`));svg.append(dot);});});
  }
  function contributionChart(svg,values){
    if(!svg)return;const {width:w,height:h}=setupSvg(svg),l=42,r=10,top=25,bottom=h-48,max=Math.max(1,...values)*1.15,slot=(w-l-r)/10;
    [0,.5,1].forEach(f=>{const y=bottom-f*(bottom-top);svg.append(svgNode('line',{x1:l,y1:y,x2:w-r,y2:y,class:'axis'}),svgNode('text',{x:l-7,y:y+4,'text-anchor':'end'},num(max*f,1)));});
    values.forEach((v,i)=>{const rect=svgNode('rect',{x:l+i*slot+2,y:bottom-v/max*(bottom-top),width:slot-4,height:v/max*(bottom-top),fill:'#a8c590',rx:2});rect.append(svgNode('title',{},`${M.diameters[i]}–${M.diameters[i+1]} mm: ${num(v)} EMD`));svg.append(rect);if(w>430||i%3===0)svg.append(svgNode('text',{x:l+(i+.5)*slot,y:bottom+20,'text-anchor':'middle'},i+1));});
    svg.append(svgNode('text',{x:l,y:14},'Kontribusi EMD'),svgNode('text',{x:(w+l-r)/2,y:h-6,'text-anchor':'middle'},'Interval diameter 1—10'));
  }
  const presets={balanced:[4,8,14,23,37,53,70,83,93,98,100],fine:[20,32,48,70,89,96,99,100,100,100,100],gravel:[1,3,6,10,17,26,39,58,80,94,100]};
  let currentTarget=presets.balanced, currentPrediction=M.cdf(Array(11).fill(0)).curve;
  if($('mass-controls'))Array.from({length:11},(_,i)=>{const label=node('label',`${num(M.diameters[i],4)} mm`),input=node('input');input.type='number';input.min=-20;input.max=20;input.step=.25;input.value=0;input.id=`logit-${i}`;input.setAttribute('aria-label',`Logit ${i+1} untuk massa hingga ${M.diameters[i]} mm`);input.addEventListener('input',updateCurve);label.append(input);$('mass-controls').append(label);});
  function updateCurve(){
    if(!$('mass-controls'))return;const logits=Array.from({length:11},(_,i)=>+$( `logit-${i}`).value);
    if(!logits.every(Number.isFinite)){put('curve-result','Isi 11 logit dengan angka valid.');return;}
    const {masses,curve}=M.cdf(logits);currentPrediction=curve;const score=M.emd(currentTarget,curve);
    cdfChart($('cdf-chart'),currentTarget,curve);contributionChart($('emd-chart'),score.contributions);
    put('curve-result',`EMD ilustrasi ${num(score.total,3)} · total massa ${num(masses.reduce((a,b)=>a+b,0))}% · endpoint ${curve[10]}% · kurva monoton`);
    const f=[curve[0],curve[3]-curve[0],curve[6]-curve[3],100-curve[6]],names=['Lempung','Lanau','Pasir','Kerikil'];$('fraction-bar').replaceChildren();
    f.forEach((v,i)=>{const seg=node('span',v>14?`${num(v,1)}%`:'');seg.style.width=`${v}%`;seg.title=`${names[i]}: ${num(v)}%`;$('fraction-bar').append(seg);});
    put('fraction-label',f.map((v,i)=>`${names[i]} ${num(v)}%`).join(' · '));
  }
  if($('soil-preset'))$('soil-preset').addEventListener('change',()=>{currentTarget=presets[$('soil-preset').value];updateCurve();});
  if($('reset-curve'))$('reset-curve').addEventListener('click',()=>{Array.from({length:11},(_,i)=>$( `logit-${i}`).value=0);updateCurve();});
  if($('download-demo'))$('download-demo').addEventListener('click',()=>{const csv=`sample_id,${M.diameters.join(',')}\nDEMO_NOT_MODEL_RESULT,${currentPrediction.map(v=>v.toFixed(4)).join(',')}\n`;const url=URL.createObjectURL(new Blob([csv],{type:'text/csv'})),a=node('a');a.href=url;a.download='illustrative_curve_NOT_submission.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
  updateCurve();cdfChart($('paper-cdf-chart'),presets.balanced,M.cdf(Array(11).fill(0)).curve);
  function updateScale(){
    if(!$('scale-chart'))return;const mm=+$('grain-mm').value,ppm=+$('camera').value,svg=$('scale-chart'),{width:w}=setupSvg(svg,210),barMax=Math.min(300,w-120),native=mm*ppm,canon=mm*10,max=Math.max(10,native,canon);
    put('grain-mm-value',`${num(mm,1)} mm`);
    [['Native',native,'#ad581c'],['Canonical',canon,'#12654f']].forEach(([label,px,color],i)=>{const y=42+i*72;svg.append(svgNode('text',{x:8,y:y-10},label),svgNode('rect',{x:8,y,width:Math.max(3,barMax*px/max),height:24,rx:4,fill:color}),svgNode('text',{x:16+barMax*px/max,y:y+17},`${num(px,1)} px`));});
    svg.append(svgNode('text',{x:8,y:194},'Panjang bar proporsional · ilustrasi diameter'));
    put('scale-result',`${num(mm,1)} mm → ${num(native,1)} px native → ${num(canon,1)} px pada 10 px/mm`);
  }
  ['camera','grain-mm'].forEach(id=>{if($(id))$(id).addEventListener('input',updateScale);});updateScale();
  function updateCost(){
    if(!$('cost-result'))return;
    try{const c=M.cost({epochs:+$('cost-epochs').value,steps:+$('cost-steps').value,folds:+$('cost-folds').value,seeds:+$('cost-seeds').value,seconds:+$('cost-seconds').value,overhead:+$('cost-overhead').value,fullFit:$('cost-full').checked}),rate=+$('cost-rate').value;if(!Number.isFinite(rate)||rate<0)throw Error('Harga harus ≥ 0');
      put('cost-result',`${num(c.updates,0)} update · ${num(c.totalHours)} jam perkiraan · $${num(c.totalHours*rate)}`);
      put('cost-detail',`Training inti ${num(c.trainingHours)} jam + overhead ${num(c.totalHours-c.trainingHours)} jam. Asumsi satu GPU, tidak termasuk antrean dan setup. Ini estimasi, bukan hasil terukur full CV.`);
    }catch(e){put('cost-result',e.message);}
  }
  document.querySelectorAll('[id^="cost-"]').forEach(el=>{if(el.matches('input'))el.addEventListener('input',updateCost);});updateCost();
  async function readJson(input){const f=input.files[0];if(!f)return null;if(f.size>20*1024*1024)throw Error('File melebihi batas 20 MiB');return JSON.parse(await f.text());}
  if($('profile-file'))$('profile-file').addEventListener('change',async e=>{try{const data=await readJson(e.target);if(!data)return;const b=data.benchmark;if(!b||!Number.isFinite(b.seconds_per_step)||b.seconds_per_step<=0)throw Error('Tidak ada benchmark seconds_per_step yang valid');
    const cfg=data.config;if(!cfg?.training||!cfg?.data)throw Error('Profile harus menyertakan konfigurasi');
    if(cfg.training.batch_size!==config.training.batch_size||cfg.training.tiles_per_sample!==config.training.tiles_per_sample||cfg.data.tile_px!==config.data.tile_px||JSON.stringify(cfg.data.fields_mm)!==JSON.stringify(config.data.fields_mm)||JSON.stringify(cfg.model)!==JSON.stringify(config.model))throw Error('Bentuk model/batch/tile/skala berbeda dari C100. Kalibrasikan profil konfigurasi yang sama.');
    $('cost-seconds').value=b.seconds_per_step;updateCost();put('profile-message',`Median terukur ${num(b.seconds_per_step,4)} detik/step pada ${b.device}; ${b.kind}. Data loader, validation dan IO belum termasuk. Nama GPU: ${b.hardware?.gpus?.[0]?.name||'CPU'}.`);
  }catch(err){put('profile-message',`Impor ditolak: ${err.message}`);}});

  function validateReport(data){
    if(data.schema_version!==1||data.evidence_type!=='complete_grouped_oof'||!Array.isArray(data.runs)||!data.runs.length||data.runs.length>100)throw Error('Schema laporan OOF tidak cocok');
    for(const r of data.runs){
      if(typeof r.experiment_id!=='string'||!Number.isFinite(r.cv_emd)||r.cv_emd<0||r.cv_emd>500||!r.ci95||![r.ci95.lower,r.ci95.upper].every(Number.isFinite)||r.ci95.lower>r.ci95.upper||!Number.isInteger(r.n_oof)||r.n_oof<1||r.n_oof>10000||!Array.isArray(r.curves)||r.curves.length!==r.n_oof)throw Error('Data run/CI/kurva tidak valid');
      const ids=new Set();for(const c of r.curves){if(typeof c.sample_id!=='string'||ids.has(c.sample_id))throw Error('ID duplikat/tidak valid');ids.add(c.sample_id);for(const key of ['target','prediction']){const a=c[key];if(!Array.isArray(a)||a.length!==11||!a.every(v=>Number.isFinite(v)&&v>=0&&v<=100)||a[10]!==100||a.some((v,i)=>i&&v<a[i-1]))throw Error('Kurva impor invalid');}}
      const score=r.curves.map(c=>M.emd(c.target,c.prediction).total).reduce((a,b)=>a+b,0)/r.n_oof;
      if(Math.abs(score-r.cv_emd)>1e-3)throw Error('Skor CV tidak cocok dengan kurva');
    }
    return data;
  }
  if($('report-file'))$('report-file').addEventListener('change',async e=>{
    try{const raw=await readJson(e.target);if(!raw)return;const data=validateReport(raw),container=$('report-content');container.replaceChildren();container.hidden=false;
      const table=node('table'),head=node('thead'),hr=node('tr');['Eksperimen','OOF','CV EMD','CI95','Group','Sumber'].forEach(v=>hr.append(node('th',v)));head.append(hr);table.append(head);const body=node('tbody');
      data.runs.forEach(r=>{const tr=node('tr');[r.experiment_id,r.n_oof,num(r.cv_emd,3),`${num(r.ci95.lower)}–${num(r.ci95.upper)}`,r.ci95.clusters??'—',(r.provenance?.source_manifest_sha256 ? `bundle ${r.provenance.source_manifest_sha256.slice(0,12)}` : r.provenance?.git_commit??'—')].forEach(v=>tr.append(node('td',String(v))));body.append(tr);});table.append(body);const wrap=node('div',undefined,'table-wrap');wrap.append(table);container.append(wrap);
      const controls=node('div',undefined,'control-row'),runLabel=node('label','Eksperimen'),runSelect=node('select'),sampleLabel=node('label','Sampel'),sampleSelect=node('select');data.runs.forEach((r,i)=>{const o=node('option',r.experiment_id);o.value=i;runSelect.append(o);});runLabel.append(runSelect);sampleLabel.append(sampleSelect);controls.append(runLabel,sampleLabel);container.append(controls);const svg=svgNode('svg',{class:'chart',role:'img','aria-label':'Target dan prediksi OOF sampel yang dipilih'}),info=node('p',undefined,'caption');container.append(svg,info);
      function sampleUpdate(){const r=data.runs[+runSelect.value],c=r.curves[+sampleSelect.value];if(!c)return;cdfChart(svg,c.target,c.prediction);const fractions=r.fraction_mae_pp;info.textContent=`${c.sample_id}: EMD ${num(M.emd(c.target,c.prediction).total,3)}. Fraction MAE run (pp): ${fractions?Object.entries(fractions).map(([k,v])=>`${k} ${num(v)}`).join(' · '):'tidak tersedia'}.`;}
      function runUpdate(){const r=data.runs[+runSelect.value];sampleSelect.replaceChildren();r.curves.forEach((c,i)=>{const o=node('option',c.sample_id);o.value=i;sampleSelect.append(o);});sampleUpdate();}
      runSelect.addEventListener('change',runUpdate);sampleSelect.addEventListener('change',sampleUpdate);runUpdate();new ResizeObserver(sampleUpdate).observe(svg);
      if(data.comparisons?.length){container.append(node('h3','Perbandingan berpasangan'));data.comparisons.forEach(c=>{if(c.delta_emd_ci95&&[c.delta_emd_ci95.estimate,c.delta_emd_ci95.lower,c.delta_emd_ci95.upper].every(Number.isFinite))container.append(node('p',`${c.candidate} − ${c.reference}: ΔEMD ${num(c.delta_emd_ci95.estimate,3)}, CI95 ${num(c.delta_emd_ci95.lower)}–${num(c.delta_emd_ci95.upper)}; kelompok membaik ${c.groups_improved}/${c.n_groups}. Delta negatif menguntungkan candidate.`,'small'));});}
      put('report-message',`${data.runs.length} run diimpor dan kurva/skor diperiksa di browser. Integritas fold diverifikasi oleh exporter Python; file ini bukan bukti autentikasi dataset. Impor tidak mengubah status global.`);put('evidence-badge','Laporan OOF diimpor');
    }catch(err){put('report-message',`Impor ditolak: ${err.message}`);$('report-content').hidden=true;}
  });
  function glossary(){if(!$('glossary-list'))return;const q=($('glossary-search')?.value||'').toLowerCase();$('glossary-list').replaceChildren();Object.entries(G).filter(([k,v])=>(k+' '+v).toLowerCase().includes(q)).sort(([a],[b])=>a.localeCompare(b)).forEach(([k,v])=>{const a=node('article');a.append(node('h3',k),node('p',v));$('glossary-list').append(a);});if(!$('glossary-list').childElementCount)$('glossary-list').append(node('p','Tidak ada istilah yang cocok.'));}
  if($('glossary-search'))$('glossary-search').addEventListener('input',glossary);glossary();
  const chartRedraw=()=>{updateCurve();updateScale();cdfChart($('paper-cdf-chart'),presets.balanced,M.cdf(Array(11).fill(0)).curve);};
  const ro=new ResizeObserver(chartRedraw);['cdf-chart','emd-chart','scale-chart','paper-cdf-chart'].forEach(id=>{if($(id))ro.observe($(id));});
  const navLinks=document.querySelectorAll('.sidebar nav a');const sectionObserver=new IntersectionObserver(entries=>{for(const e of entries)if(e.isIntersecting){navLinks.forEach(a=>a.classList.toggle('active',a.hash===`#${e.target.id}`));}},{rootMargin:'-10% 0px -65% 0px'});document.querySelectorAll('section[id]').forEach(s=>sectionObserver.observe(s));
})();
