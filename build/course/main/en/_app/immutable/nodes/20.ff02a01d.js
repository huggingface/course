import{s as le,o as oe,n as re}from"../chunks/scheduler.37c15a92.js";import{S as fe,i as pe,g as f,s,r as C,A as me,h as p,f as n,c as l,j as ae,u as S,x as M,k as Q,y as ue,a,v as F,d as P,t as y,w as R,m as ie,n as se}from"../chunks/index.2bf4358c.js";import{T as ce}from"../chunks/Tip.363c041f.js";import{H}from"../chunks/Heading.8ada512a.js";function he(E){let o,r,u="creating an account",m;return{c(){o=ie("⚠️ In order to benefit from all features available with the Model Hub and 🤗 Transformers, we recommend "),r=f("a"),r.textContent=u,m=ie("."),this.h()},l(i){o=se(i,"⚠️ In order to benefit from all features available with the Model Hub and 🤗 Transformers, we recommend "),r=p(i,"A",{href:!0,"data-svelte-h":!0}),M(r)!=="svelte-x4yw9l"&&(r.textContent=u),m=se(i,"."),this.h()},h(){Q(r,"href","https://huggingface.co/join")},m(i,c){a(i,o,c),a(i,r,c),a(i,m,c)},p:re,d(i){i&&(n(o),n(r),n(m))}}}function ge(E){let o,r,u,m,i,c,g,V='In <a href="/course/chapter2/2">Chapter 2 Section 2</a>, we saw that generative language models can be fine-tuned on specific tasks like summarization and question answering. However, nowadays it is far more common to fine-tune language models on a broad range of tasks simultaneously; a method known as supervised fine-tuning (SFT). This process helps models become more versatile and capable of handling diverse use cases. Most LLMs that people interact with on platforms like ChatGPT have undergone SFT to make them more helpful and aligned with human preferences. We will separate this chapter into four sections:',z,d,I,$,X="Chat templates structure interactions between users and AI models, ensuring consistent and contextually appropriate responses. They include components like system prompts and role-based messages.",j,w,q,v,Y='Supervised Fine-Tuning (SFT) is a critical process for adapting pre-trained language models to specific tasks. It involves training the model on a task-specific dataset with labeled examples. For a detailed guide on SFT, including key steps and best practices, see <a href="https://huggingface.co/docs/trl/en/sft_trainer" rel="nofollow">the supervised fine-tuning section of the TRL documentation</a>.',G,T,O,_,Z="Low Rank Adaptation (LoRA) is a technique for fine-tuning language models by adding low-rank matrices to the model’s layers. This allows for efficient fine-tuning while preserving the model’s pre-trained knowledge. One of the key benefits of LoRA is the significant memory savings it offers, making it possible to fine-tune large models on hardware with limited resources.",U,x,D,b,ee="Evaluation is a crucial step in the fine-tuning process. It allows us to measure the performance of the model on a task-specific dataset.",J,h,N,L,W,k,te='<li><a href="https://huggingface.co/docs/transformers/main/en/chat_templating" rel="nofollow">Transformers documentation on chat templates</a></li> <li><a href="https://github.com/huggingface/trl/blob/main/trl/scripts/sft.py" rel="nofollow">Script for Supervised Fine-Tuning in TRL</a></li> <li><a href="https://huggingface.co/docs/trl/main/en/sft_trainer" rel="nofollow"><code>SFTTrainer</code> in TRL</a></li> <li><a href="https://arxiv.org/abs/2305.18290" rel="nofollow">Direct Preference Optimization Paper</a></li> <li><a href="https://huggingface.co/docs/trl/sft_trainer" rel="nofollow">Supervised Fine-Tuning with TRL</a></li> <li><a href="https://github.com/huggingface/alignment-handbook" rel="nofollow">How to fine-tune Google Gemma with ChatML and Hugging Face TRL</a></li> <li><a href="https://huggingface.co/learn/cookbook/en/fine_tuning_llm_to_generate_persian_product_catalogs_in_json_format" rel="nofollow">Fine-tuning LLM to Generate Persian Product Catalogs in JSON Format</a></li>',B,A,K;return i=new H({props:{title:"Supervised Fine-Tuning",local:"supervised-fine-tuning",headingTag:"h1"}}),d=new H({props:{title:"1️⃣ Chat Templates",local:"1-chat-templates",headingTag:"h2"}}),w=new H({props:{title:"2️⃣ Supervised Fine-Tuning",local:"2-supervised-fine-tuning",headingTag:"h2"}}),T=new H({props:{title:"3️⃣ Low Rank Adaptation (LoRA)",local:"3-low-rank-adaptation-lora",headingTag:"h2"}}),x=new H({props:{title:"4️⃣ Evaluation",local:"4-evaluation",headingTag:"h2"}}),h=new ce({props:{$$slots:{default:[he]},$$scope:{ctx:E}}}),L=new H({props:{title:"References",local:"references",headingTag:"h2"}}),{c(){o=f("meta"),r=s(),u=f("p"),m=s(),C(i.$$.fragment),c=s(),g=f("p"),g.innerHTML=V,z=s(),C(d.$$.fragment),I=s(),$=f("p"),$.textContent=X,j=s(),C(w.$$.fragment),q=s(),v=f("p"),v.innerHTML=Y,G=s(),C(T.$$.fragment),O=s(),_=f("p"),_.textContent=Z,U=s(),C(x.$$.fragment),D=s(),b=f("p"),b.textContent=ee,J=s(),C(h.$$.fragment),N=s(),C(L.$$.fragment),W=s(),k=f("ul"),k.innerHTML=te,B=s(),A=f("p"),this.h()},l(e){const t=me("svelte-u9bgzb",document.head);o=p(t,"META",{name:!0,content:!0}),t.forEach(n),r=l(e),u=p(e,"P",{}),ae(u).forEach(n),m=l(e),S(i.$$.fragment,e),c=l(e),g=p(e,"P",{"data-svelte-h":!0}),M(g)!=="svelte-1xnxhzs"&&(g.innerHTML=V),z=l(e),S(d.$$.fragment,e),I=l(e),$=p(e,"P",{"data-svelte-h":!0}),M($)!=="svelte-1z0l1dz"&&($.textContent=X),j=l(e),S(w.$$.fragment,e),q=l(e),v=p(e,"P",{"data-svelte-h":!0}),M(v)!=="svelte-16r93j"&&(v.innerHTML=Y),G=l(e),S(T.$$.fragment,e),O=l(e),_=p(e,"P",{"data-svelte-h":!0}),M(_)!=="svelte-1sh6um3"&&(_.textContent=Z),U=l(e),S(x.$$.fragment,e),D=l(e),b=p(e,"P",{"data-svelte-h":!0}),M(b)!=="svelte-i6pwpj"&&(b.textContent=ee),J=l(e),S(h.$$.fragment,e),N=l(e),S(L.$$.fragment,e),W=l(e),k=p(e,"UL",{"data-svelte-h":!0}),M(k)!=="svelte-qd9tgd"&&(k.innerHTML=te),B=l(e),A=p(e,"P",{}),ae(A).forEach(n),this.h()},h(){Q(o,"name","hf:doc:metadata"),Q(o,"content",de)},m(e,t){ue(document.head,o),a(e,r,t),a(e,u,t),a(e,m,t),F(i,e,t),a(e,c,t),a(e,g,t),a(e,z,t),F(d,e,t),a(e,I,t),a(e,$,t),a(e,j,t),F(w,e,t),a(e,q,t),a(e,v,t),a(e,G,t),F(T,e,t),a(e,O,t),a(e,_,t),a(e,U,t),F(x,e,t),a(e,D,t),a(e,b,t),a(e,J,t),F(h,e,t),a(e,N,t),F(L,e,t),a(e,W,t),a(e,k,t),a(e,B,t),a(e,A,t),K=!0},p(e,[t]){const ne={};t&2&&(ne.$$scope={dirty:t,ctx:e}),h.$set(ne)},i(e){K||(P(i.$$.fragment,e),P(d.$$.fragment,e),P(w.$$.fragment,e),P(T.$$.fragment,e),P(x.$$.fragment,e),P(h.$$.fragment,e),P(L.$$.fragment,e),K=!0)},o(e){y(i.$$.fragment,e),y(d.$$.fragment,e),y(w.$$.fragment,e),y(T.$$.fragment,e),y(x.$$.fragment,e),y(h.$$.fragment,e),y(L.$$.fragment,e),K=!1},d(e){e&&(n(r),n(u),n(m),n(c),n(g),n(z),n(I),n($),n(j),n(q),n(v),n(G),n(O),n(_),n(U),n(D),n(b),n(J),n(N),n(W),n(k),n(B),n(A)),n(o),R(i,e),R(d,e),R(w,e),R(T,e),R(x,e),R(h,e),R(L,e)}}}const de='{"title":"Supervised Fine-Tuning","local":"supervised-fine-tuning","sections":[{"title":"1️⃣ Chat Templates","local":"1-chat-templates","sections":[],"depth":2},{"title":"2️⃣ Supervised Fine-Tuning","local":"2-supervised-fine-tuning","sections":[],"depth":2},{"title":"3️⃣ Low Rank Adaptation (LoRA)","local":"3-low-rank-adaptation-lora","sections":[],"depth":2},{"title":"4️⃣ Evaluation","local":"4-evaluation","sections":[],"depth":2},{"title":"References","local":"references","sections":[],"depth":2}],"depth":1}';function $e(E){return oe(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class xe extends fe{constructor(o){super(),pe(this,o,$e,ge,le,{})}}export{xe as component};
