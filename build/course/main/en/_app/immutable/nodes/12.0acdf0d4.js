import{s as z,n as I,o as j}from"../chunks/scheduler.37c15a92.js";import{S as D,i as G,g as u,s as d,r as R,A as N,h,f as a,c,j as A,u as k,x,k as B,y as F,a as n,v as M,d as H,t as S,w as q}from"../chunks/index.2bf4358c.js";import{C as U}from"../chunks/CourseFloatingBanner.15ba07e6.js";import{H as W}from"../chunks/Heading.8ada512a.js";function X(b){let s,T,f,w,o,y,r,_,i,P="In this chapter, you saw how to approach different NLP tasks using the high-level <code>pipeline()</code> function from 🤗 Transformers. You also saw how to search for and use models in the Hub, as well as how to use the Inference API to test the models directly in your browser.",g,l,L="We discussed how Transformer models work at a high level, and talked about the importance of transfer learning and fine-tuning. A key aspect is that you can use the full architecture or only the encoder or decoder, depending on what kind of task you aim to solve. The following table summarizes this:",$,m,C="<thead><tr><th>Model</th> <th>Examples</th> <th>Tasks</th></tr></thead> <tbody><tr><td>Encoder</td> <td>ALBERT, BERT, DistilBERT, ELECTRA, RoBERTa</td> <td>Sentence classification, named entity recognition, extractive question answering</td></tr> <tr><td>Decoder</td> <td>CTRL, GPT, GPT-2, Transformer XL</td> <td>Text generation</td></tr> <tr><td>Encoder-decoder</td> <td>BART, T5, Marian, mBART</td> <td>Summarization, translation, generative question answering</td></tr></tbody>",v,p,E;return o=new W({props:{title:"Summary",local:"summary",headingTag:"h1"}}),r=new U({props:{chapter:1,classNames:"absolute z-10 right-0 top-0"}}),{c(){s=u("meta"),T=d(),f=u("p"),w=d(),R(o.$$.fragment),y=d(),R(r.$$.fragment),_=d(),i=u("p"),i.innerHTML=P,g=d(),l=u("p"),l.textContent=L,$=d(),m=u("table"),m.innerHTML=C,v=d(),p=u("p"),this.h()},l(t){const e=N("svelte-u9bgzb",document.head);s=h(e,"META",{name:!0,content:!0}),e.forEach(a),T=c(t),f=h(t,"P",{}),A(f).forEach(a),w=c(t),k(o.$$.fragment,t),y=c(t),k(r.$$.fragment,t),_=c(t),i=h(t,"P",{"data-svelte-h":!0}),x(i)!=="svelte-1hpqyzf"&&(i.innerHTML=P),g=c(t),l=h(t,"P",{"data-svelte-h":!0}),x(l)!=="svelte-xp4tj6"&&(l.textContent=L),$=c(t),m=h(t,"TABLE",{"data-svelte-h":!0}),x(m)!=="svelte-1v7q8ve"&&(m.innerHTML=C),v=c(t),p=h(t,"P",{}),A(p).forEach(a),this.h()},h(){B(s,"name","hf:doc:metadata"),B(s,"content",Y)},m(t,e){F(document.head,s),n(t,T,e),n(t,f,e),n(t,w,e),M(o,t,e),n(t,y,e),M(r,t,e),n(t,_,e),n(t,i,e),n(t,g,e),n(t,l,e),n(t,$,e),n(t,m,e),n(t,v,e),n(t,p,e),E=!0},p:I,i(t){E||(H(o.$$.fragment,t),H(r.$$.fragment,t),E=!0)},o(t){S(o.$$.fragment,t),S(r.$$.fragment,t),E=!1},d(t){t&&(a(T),a(f),a(w),a(y),a(_),a(i),a(g),a(l),a($),a(m),a(v),a(p)),a(s),q(o,t),q(r,t)}}}const Y='{"title":"Summary","local":"summary","sections":[],"depth":1}';function J(b){return j(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Z extends D{constructor(s){super(),G(this,s,J,X,z,{})}}export{Z as component};
