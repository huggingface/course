import{s as st,o as at,n as it}from"../chunks/scheduler.37c15a92.js";import{S as ot,i as rt,g as i,s,r as u,A as pt,h as o,f as n,c as a,j as nt,u as m,x as r,k as lt,y as ut,a as l,v as h,d as f,t as c,w as y}from"../chunks/index.2bf4358c.js";import{T as mt}from"../chunks/Tip.363c041f.js";import{Y as ht}from"../chunks/Youtube.1e50a667.js";import{C as He}from"../chunks/CodeBlock.4f5fc1ad.js";import{C as ft}from"../chunks/CourseFloatingBanner.15ba07e6.js";import{H as b}from"../chunks/Heading.8ada512a.js";function ct(ee){let p,w="🚨 Many issues in the 🤗 Transformers repository are unsolved because the data used to reproduce them is not accessible.";return{c(){p=i("p"),p.textContent=w},l(d){p=o(d,"P",{"data-svelte-h":!0}),r(p)!=="svelte-18vj86w"&&(p.textContent=w)},m(d,q){l(d,p,q)},p:it,d(d){d&&n(p)}}}function yt(ee){let p,w,d,q,$,te,v,ne,x,Ge='When you encounter something that doesn’t seem right with one of the Hugging Face libraries, you should definitely let us know so we can fix it (the same goes for any open source library, for that matter). If you are not completely certain whether the bug lies in your own code or one of our libraries, the first place to check is the <a href="https://discuss.huggingface.co/" rel="nofollow">forums</a>. The community will help you figure this out, and the Hugging Face team also closely watches the discussions there.',le,j,se,M,Ne="When you are sure you have a bug in your hand, the first step is to build a minimal reproducible example.",ae,T,ie,k,Ee="It’s very important to isolate the piece of code that produces the bug, as no one in the Hugging Face team is a magician (yet), and they can’t fix what they can’t see. A minimal reproducible example should, as the name indicates, be reproducible. This means that it should not rely on any external files or data you may have. Try to replace the data you are using with some dummy values that look like your real ones and still produce the same error.",oe,g,re,C,ze="Once you have something that is self-contained, you can try to reduce it into even less lines of code, building what we call a <em>minimal reproducible example</em>. While this requires a bit more work on your side, you will almost be guaranteed to get help and a fix if you provide a nice, short bug reproducer.",pe,U,Se="If you feel comfortable enough, go inspect the source code where your bug happens. You might find a solution to your problem (in which case you can even suggest a pull request to fix it), but more generally, this can help the maintainers better understand the source when they read your report.",ue,J,me,_,Fe='When you file your issue, you will notice there is a template to fill out. We will follow the one for <a href="https://github.com/huggingface/transformers/issues/new/choose" rel="nofollow">🤗 Transformers issues</a> here, but the same kind of information will be required if you report an issue in another repository. Don’t leave the template blank: taking the time to fill it in will maximize your chances of getting an answer and solving your problem.',he,I,Ye='In general, when filing an issue, always stay courteous. This is an open source project, so you are using free software, and no one has any obligation to help you. You may include what you feel is justified criticism in your issue, but then the maintainers may very well take it badly and not be in a rush help you. Make sure you read the <a href="https://github.com/huggingface/transformers/blob/master/CODE_OF_CONDUCT.md" rel="nofollow">code of conduct</a> of the project.',fe,B,ce,P,Ze="🤗 Transformers provides a utility to get all the information we need about your environment. Just type the following in your terminal:",ye,L,de,H,Ae="and you should get something like this:",ge,G,be,N,Qe="You can also add a <code>!</code> at the beginning of the <code>transformers-cli env</code> command to execute it from a notebook cell, and then copy and paste the result at the beginning of your issue.",we,E,$e,z,Ve="Tagging people by typing an <code>@</code> followed by their GitHub handle will send them a notification so they will see your issue and might reply quicker. Use this with moderation, because the people you tag might not appreciate being notified if it’s something they have no direct link to. If you have looked at the source files related to your bug, you should tag the last person that made changes at the line you think is responsible for your problem (you can find this information by looking at said line on GitHub, selecting it, then clicking “View git blame”).",ve,S,We="Otherwise, the template offers suggestions of people to tag. In general, never tag more than three people!",xe,F,je,Y,Xe="If you have managed to create a self-contained example that produces the bug, now is the time to include it! Type a line with three backticks followed by <code>python</code>, like this:",Me,Z,Te,A,Re="then paste in your minimal reproducible example and type a new line with three backticks. This will ensure your code is properly formatted.",ke,Q,De="If you didn’t manage to create a reproducible example, explain in clear steps how you got to your issue. Include a link to a Google Colab notebook where you got the error if you can. The more information you share, the better able the maintainers will be to reply to you.",Ce,V,Oe="In all cases, you should copy and paste the whole error message you are getting. If you’re working in Colab, remember that some of the frames may be automatically collapsed in the stack trace, so make sure you expand them before copying. Like with the code sample, put that error message between two lines with three backticks, so it’s properly formatted.",Ue,W,Je,X,qe="Explain in a few lines what you expected to get, so that the maintainers get a full grasp of the problem. This part is generally pretty obvious, so it should fit in one sentence, but in some cases you may have a lot to say.",_e,R,Ie,D,Ke="Once your issue is filed, make sure to quickly check everything looks okay. You can edit the issue if you made a mistake, or even change its title if you realize the problem is different from what you initially thought.",Be,O,et="There is no point pinging people if you don’t get an answer. If no one helps you in a few days, it’s likely that no one could make sense of your problem. Don’t hesitate to go back to the reproducible example. Can you make it shorter and more to the point? If you don’t get an answer in a week, you can leave a message gently asking for help, especially if you’ve edited your issue to include more information on the problem.",Pe,K,Le;return $=new b({props:{title:"How to write a good issue",local:"how-to-write-a-good-issue",headingTag:"h1"}}),v=new ft({props:{chapter:8,classNames:"absolute z-10 right-0 top-0",notebooks:[{label:"Google Colab",value:"https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter8/section5.ipynb"},{label:"Aws Studio",value:"https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter8/section5.ipynb"}]}}),j=new ht({props:{id:"_PAli-V4wj0"}}),T=new b({props:{title:"Creating a minimal reproducible example",local:"creating-a-minimal-reproducible-example",headingTag:"h2"}}),g=new mt({props:{$$slots:{default:[ct]},$$scope:{ctx:ee}}}),J=new b({props:{title:"Filling out the issue template",local:"filling-out-the-issue-template",headingTag:"h2"}}),B=new b({props:{title:"Including your environment information",local:"including-your-environment-information",headingTag:"h3"}}),L=new He({props:{code:"dHJhbnNmb3JtZXJzLWNsaSUyMGVudg==",highlighted:'transformers-<span class="hljs-keyword">cli</span> env',wrap:!1}}),G=new He({props:{code:"Q29weS1hbmQtcGFzdGUlMjB0aGUlMjB0ZXh0JTIwYmVsb3clMjBpbiUyMHlvdXIlMjBHaXRIdWIlMjBpc3N1ZSUyMGFuZCUyMEZJTEwlMjBPVVQlMjB0aGUlMjB0d28lMjBsYXN0JTIwcG9pbnRzLiUwQSUwQS0lMjAlNjB0cmFuc2Zvcm1lcnMlNjAlMjB2ZXJzaW9uJTNBJTIwNC4xMi4wLmRldjAlMEEtJTIwUGxhdGZvcm0lM0ElMjBMaW51eC01LjEwLjYxLTEtTUFOSkFSTy14ODZfNjQtd2l0aC1hcmNoLU1hbmphcm8tTGludXglMEEtJTIwUHl0aG9uJTIwdmVyc2lvbiUzQSUyMDMuNy45JTBBLSUyMFB5VG9yY2glMjB2ZXJzaW9uJTIwKEdQVSUzRiklM0ElMjAxLjguMSUyQmN1MTExJTIwKFRydWUpJTBBLSUyMFRlbnNvcmZsb3clMjB2ZXJzaW9uJTIwKEdQVSUzRiklM0ElMjAyLjUuMCUyMChUcnVlKSUwQS0lMjBGbGF4JTIwdmVyc2lvbiUyMChDUFUlM0YlMkZHUFUlM0YlMkZUUFUlM0YpJTNBJTIwMC4zLjQlMjAoY3B1KSUwQS0lMjBKYXglMjB2ZXJzaW9uJTNBJTIwMC4yLjEzJTBBLSUyMEpheExpYiUyMHZlcnNpb24lM0ElMjAwLjEuNjUlMEEtJTIwVXNpbmclMjBHUFUlMjBpbiUyMHNjcmlwdCUzRiUzQSUyMCUzQ2ZpbGwlMjBpbiUzRSUwQS0lMjBVc2luZyUyMGRpc3RyaWJ1dGVkJTIwb3IlMjBwYXJhbGxlbCUyMHNldC11cCUyMGluJTIwc2NyaXB0JTNGJTNBJTIwJTNDZmlsbCUyMGluJTNF",highlighted:`<span class="hljs-keyword">Copy</span>-<span class="hljs-keyword">and</span>-paste the <span class="hljs-type">text</span> below <span class="hljs-keyword">in</span> your GitHub issue <span class="hljs-keyword">and</span> FILL <span class="hljs-keyword">OUT</span> the two last points.

- \`transformers\` <span class="hljs-keyword">version</span>: <span class="hljs-number">4.12</span><span class="hljs-number">.0</span>.dev0
- Platform: Linux<span class="hljs-number">-5.10</span><span class="hljs-number">.61</span><span class="hljs-number">-1</span>-MANJARO-x86_64-<span class="hljs-keyword">with</span>-arch-Manjaro-Linux
- Python <span class="hljs-keyword">version</span>: <span class="hljs-number">3.7</span><span class="hljs-number">.9</span>
- PyTorch version (GPU?): <span class="hljs-number">1.8</span><span class="hljs-number">.1</span>+cu111 (<span class="hljs-keyword">True</span>)
- Tensorflow version (GPU?): <span class="hljs-number">2.5</span><span class="hljs-number">.0</span> (<span class="hljs-keyword">True</span>)
- Flax version (CPU?/GPU?/TPU?): <span class="hljs-number">0.3</span><span class="hljs-number">.4</span> (cpu)
- Jax <span class="hljs-keyword">version</span>: <span class="hljs-number">0.2</span><span class="hljs-number">.13</span>
- JaxLib <span class="hljs-keyword">version</span>: <span class="hljs-number">0.1</span><span class="hljs-number">.65</span>
- <span class="hljs-keyword">Using</span> GPU <span class="hljs-keyword">in</span> script?: &lt;fill <span class="hljs-keyword">in</span>&gt;
- <span class="hljs-keyword">Using</span> distributed <span class="hljs-keyword">or</span> parallel <span class="hljs-keyword">set</span>-up <span class="hljs-keyword">in</span> script?: &lt;fill <span class="hljs-keyword">in</span>&gt;`,wrap:!1}}),E=new b({props:{title:"Tagging people",local:"tagging-people",headingTag:"h3"}}),F=new b({props:{title:"Including a reproducible example",local:"including-a-reproducible-example",headingTag:"h3"}}),Z=new He({props:{code:"JTYwJTYwJTYwcHl0aG9u",highlighted:"```python",wrap:!1}}),W=new b({props:{title:"Describing the expected behavior",local:"describing-the-expected-behavior",headingTag:"h3"}}),R=new b({props:{title:"And then what?",local:"and-then-what",headingTag:"h2"}}),{c(){p=i("meta"),w=s(),d=i("p"),q=s(),u($.$$.fragment),te=s(),u(v.$$.fragment),ne=s(),x=i("p"),x.innerHTML=Ge,le=s(),u(j.$$.fragment),se=s(),M=i("p"),M.textContent=Ne,ae=s(),u(T.$$.fragment),ie=s(),k=i("p"),k.textContent=Ee,oe=s(),u(g.$$.fragment),re=s(),C=i("p"),C.innerHTML=ze,pe=s(),U=i("p"),U.textContent=Se,ue=s(),u(J.$$.fragment),me=s(),_=i("p"),_.innerHTML=Fe,he=s(),I=i("p"),I.innerHTML=Ye,fe=s(),u(B.$$.fragment),ce=s(),P=i("p"),P.textContent=Ze,ye=s(),u(L.$$.fragment),de=s(),H=i("p"),H.textContent=Ae,ge=s(),u(G.$$.fragment),be=s(),N=i("p"),N.innerHTML=Qe,we=s(),u(E.$$.fragment),$e=s(),z=i("p"),z.innerHTML=Ve,ve=s(),S=i("p"),S.textContent=We,xe=s(),u(F.$$.fragment),je=s(),Y=i("p"),Y.innerHTML=Xe,Me=s(),u(Z.$$.fragment),Te=s(),A=i("p"),A.textContent=Re,ke=s(),Q=i("p"),Q.textContent=De,Ce=s(),V=i("p"),V.textContent=Oe,Ue=s(),u(W.$$.fragment),Je=s(),X=i("p"),X.textContent=qe,_e=s(),u(R.$$.fragment),Ie=s(),D=i("p"),D.textContent=Ke,Be=s(),O=i("p"),O.textContent=et,Pe=s(),K=i("p"),this.h()},l(e){const t=pt("svelte-u9bgzb",document.head);p=o(t,"META",{name:!0,content:!0}),t.forEach(n),w=a(e),d=o(e,"P",{}),nt(d).forEach(n),q=a(e),m($.$$.fragment,e),te=a(e),m(v.$$.fragment,e),ne=a(e),x=o(e,"P",{"data-svelte-h":!0}),r(x)!=="svelte-eufcd4"&&(x.innerHTML=Ge),le=a(e),m(j.$$.fragment,e),se=a(e),M=o(e,"P",{"data-svelte-h":!0}),r(M)!=="svelte-ptniez"&&(M.textContent=Ne),ae=a(e),m(T.$$.fragment,e),ie=a(e),k=o(e,"P",{"data-svelte-h":!0}),r(k)!=="svelte-1uqbgpe"&&(k.textContent=Ee),oe=a(e),m(g.$$.fragment,e),re=a(e),C=o(e,"P",{"data-svelte-h":!0}),r(C)!=="svelte-m1mh43"&&(C.innerHTML=ze),pe=a(e),U=o(e,"P",{"data-svelte-h":!0}),r(U)!=="svelte-1qvewmp"&&(U.textContent=Se),ue=a(e),m(J.$$.fragment,e),me=a(e),_=o(e,"P",{"data-svelte-h":!0}),r(_)!=="svelte-p88gaz"&&(_.innerHTML=Fe),he=a(e),I=o(e,"P",{"data-svelte-h":!0}),r(I)!=="svelte-183br84"&&(I.innerHTML=Ye),fe=a(e),m(B.$$.fragment,e),ce=a(e),P=o(e,"P",{"data-svelte-h":!0}),r(P)!=="svelte-z3txoy"&&(P.textContent=Ze),ye=a(e),m(L.$$.fragment,e),de=a(e),H=o(e,"P",{"data-svelte-h":!0}),r(H)!=="svelte-ris3ty"&&(H.textContent=Ae),ge=a(e),m(G.$$.fragment,e),be=a(e),N=o(e,"P",{"data-svelte-h":!0}),r(N)!=="svelte-1kg5wcv"&&(N.innerHTML=Qe),we=a(e),m(E.$$.fragment,e),$e=a(e),z=o(e,"P",{"data-svelte-h":!0}),r(z)!=="svelte-3emejn"&&(z.innerHTML=Ve),ve=a(e),S=o(e,"P",{"data-svelte-h":!0}),r(S)!=="svelte-11lklvz"&&(S.textContent=We),xe=a(e),m(F.$$.fragment,e),je=a(e),Y=o(e,"P",{"data-svelte-h":!0}),r(Y)!=="svelte-1mo97uw"&&(Y.innerHTML=Xe),Me=a(e),m(Z.$$.fragment,e),Te=a(e),A=o(e,"P",{"data-svelte-h":!0}),r(A)!=="svelte-1cb4wst"&&(A.textContent=Re),ke=a(e),Q=o(e,"P",{"data-svelte-h":!0}),r(Q)!=="svelte-8z9sr1"&&(Q.textContent=De),Ce=a(e),V=o(e,"P",{"data-svelte-h":!0}),r(V)!=="svelte-1nzy10a"&&(V.textContent=Oe),Ue=a(e),m(W.$$.fragment,e),Je=a(e),X=o(e,"P",{"data-svelte-h":!0}),r(X)!=="svelte-1decsmy"&&(X.textContent=qe),_e=a(e),m(R.$$.fragment,e),Ie=a(e),D=o(e,"P",{"data-svelte-h":!0}),r(D)!=="svelte-ln4y6s"&&(D.textContent=Ke),Be=a(e),O=o(e,"P",{"data-svelte-h":!0}),r(O)!=="svelte-81ssmx"&&(O.textContent=et),Pe=a(e),K=o(e,"P",{}),nt(K).forEach(n),this.h()},h(){lt(p,"name","hf:doc:metadata"),lt(p,"content",dt)},m(e,t){ut(document.head,p),l(e,w,t),l(e,d,t),l(e,q,t),h($,e,t),l(e,te,t),h(v,e,t),l(e,ne,t),l(e,x,t),l(e,le,t),h(j,e,t),l(e,se,t),l(e,M,t),l(e,ae,t),h(T,e,t),l(e,ie,t),l(e,k,t),l(e,oe,t),h(g,e,t),l(e,re,t),l(e,C,t),l(e,pe,t),l(e,U,t),l(e,ue,t),h(J,e,t),l(e,me,t),l(e,_,t),l(e,he,t),l(e,I,t),l(e,fe,t),h(B,e,t),l(e,ce,t),l(e,P,t),l(e,ye,t),h(L,e,t),l(e,de,t),l(e,H,t),l(e,ge,t),h(G,e,t),l(e,be,t),l(e,N,t),l(e,we,t),h(E,e,t),l(e,$e,t),l(e,z,t),l(e,ve,t),l(e,S,t),l(e,xe,t),h(F,e,t),l(e,je,t),l(e,Y,t),l(e,Me,t),h(Z,e,t),l(e,Te,t),l(e,A,t),l(e,ke,t),l(e,Q,t),l(e,Ce,t),l(e,V,t),l(e,Ue,t),h(W,e,t),l(e,Je,t),l(e,X,t),l(e,_e,t),h(R,e,t),l(e,Ie,t),l(e,D,t),l(e,Be,t),l(e,O,t),l(e,Pe,t),l(e,K,t),Le=!0},p(e,[t]){const tt={};t&2&&(tt.$$scope={dirty:t,ctx:e}),g.$set(tt)},i(e){Le||(f($.$$.fragment,e),f(v.$$.fragment,e),f(j.$$.fragment,e),f(T.$$.fragment,e),f(g.$$.fragment,e),f(J.$$.fragment,e),f(B.$$.fragment,e),f(L.$$.fragment,e),f(G.$$.fragment,e),f(E.$$.fragment,e),f(F.$$.fragment,e),f(Z.$$.fragment,e),f(W.$$.fragment,e),f(R.$$.fragment,e),Le=!0)},o(e){c($.$$.fragment,e),c(v.$$.fragment,e),c(j.$$.fragment,e),c(T.$$.fragment,e),c(g.$$.fragment,e),c(J.$$.fragment,e),c(B.$$.fragment,e),c(L.$$.fragment,e),c(G.$$.fragment,e),c(E.$$.fragment,e),c(F.$$.fragment,e),c(Z.$$.fragment,e),c(W.$$.fragment,e),c(R.$$.fragment,e),Le=!1},d(e){e&&(n(w),n(d),n(q),n(te),n(ne),n(x),n(le),n(se),n(M),n(ae),n(ie),n(k),n(oe),n(re),n(C),n(pe),n(U),n(ue),n(me),n(_),n(he),n(I),n(fe),n(ce),n(P),n(ye),n(de),n(H),n(ge),n(be),n(N),n(we),n($e),n(z),n(ve),n(S),n(xe),n(je),n(Y),n(Me),n(Te),n(A),n(ke),n(Q),n(Ce),n(V),n(Ue),n(Je),n(X),n(_e),n(Ie),n(D),n(Be),n(O),n(Pe),n(K)),n(p),y($,e),y(v,e),y(j,e),y(T,e),y(g,e),y(J,e),y(B,e),y(L,e),y(G,e),y(E,e),y(F,e),y(Z,e),y(W,e),y(R,e)}}}const dt='{"title":"How to write a good issue","local":"how-to-write-a-good-issue","sections":[{"title":"Creating a minimal reproducible example","local":"creating-a-minimal-reproducible-example","sections":[],"depth":2},{"title":"Filling out the issue template","local":"filling-out-the-issue-template","sections":[{"title":"Including your environment information","local":"including-your-environment-information","sections":[],"depth":3},{"title":"Tagging people","local":"tagging-people","sections":[],"depth":3},{"title":"Including a reproducible example","local":"including-a-reproducible-example","sections":[],"depth":3},{"title":"Describing the expected behavior","local":"describing-the-expected-behavior","sections":[],"depth":3}],"depth":2},{"title":"And then what?","local":"and-then-what","sections":[],"depth":2}],"depth":1}';function gt(ee){return at(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Tt extends ot{constructor(p){super(),rt(this,p,gt,yt,st,{})}}export{Tt as component};
