/* Shared numerical definitions. All demo curves are illustrative, not results. */
(function(root) {
  'use strict';
  const diameters = [0.002,0.0063,0.02,0.063,0.2,0.63,2,6.3,20,63,200];
  const weights = diameters.slice(1).map((d,i)=>Math.log10(d)-Math.log10(diameters[i]));
  function cdf(logits) {
    if(logits.length!==11 || !logits.every(Number.isFinite)) throw Error('11 logit finite diperlukan');
    const max=Math.max(...logits), exp=logits.map(v=>Math.exp(v-max)), sum=exp.reduce((a,b)=>a+b,0);
    const masses=exp.map(v=>100*v/sum); let total=0;
    const curve=masses.map(v=>total+=v); curve[10]=100;
    return {masses,curve};
  }
  function emd(a,b) {
    if(a.length!==11 || b.length!==11 || ![...a,...b].every(Number.isFinite)) throw Error('Kurva harus memiliki 11 nilai finite');
    const contributions=weights.map((w,i)=>w*Math.abs(a[i]-b[i]));
    return {total:contributions.reduce((a,b)=>a+b,0),contributions};
  }
  function cost({epochs,steps,folds,seeds,seconds,overhead,fullFit=false}) {
    if(![epochs,steps,folds,seeds,seconds,overhead].every(Number.isFinite)||Math.min(epochs,steps,folds,seeds,seconds)<=0||overhead<0) throw Error('Parameter biaya tidak valid');
    const updates=epochs*steps*(folds+(fullFit?1:0))*seeds;
    return {updates,trainingHours:updates*seconds/3600,totalHours:updates*seconds/3600*(1+overhead/100)};
  }
  const api={diameters,weights,cdf,emd,cost}; root.PhyMath=api;
  if(typeof module!=='undefined') module.exports=api;
})(typeof globalThis!=='undefined'?globalThis:this);
