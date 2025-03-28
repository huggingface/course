import{s as je,o as Ee}from"../chunks/scheduler.37c15a92.js";import{S as Fe,i as Se,g as qe,s as o,r as h,A as Ae,h as Te,f as a,c as i,j as Be,u as p,x as Ie,k as Ne,y as Pe,a as s,v as u,t as r,b as He,d as l,w as m,p as Ye}from"../chunks/index.2bf4358c.js";import{C as Le}from"../chunks/CourseFloatingBanner.15ba07e6.js";import{Q as w}from"../chunks/Question.668688bc.js";import{F as Me}from"../chunks/FrameworkSwitchCourse.8d4d4ab6.js";import{H as $}from"../chunks/Heading.8ada512a.js";function Qe(k){let c,d,f,g;return c=new $({props:{title:"9. Why is it often unnecessary to specify a loss when calling compile() on a Transformer model?",local:"9-why-is-it-often-unnecessary-to-specify-a-loss-when-calling-compile-on-a-transformer-model",headingTag:"h3"}}),f=new w({props:{choices:[{text:"Because Transformer models are trained with unsupervised learning",explain:"Not quite -- even unsupervised learning needs a loss function!"},{text:"Because the model's internal loss output is used by default",explain:"That's correct!",correct:!0},{text:"Because we compute metrics after training instead",explain:"We do often do that, but it doesn't explain where we get the loss value we optimize in training."},{text:"Because loss is specified in `model.fit()` instead",explain:"No, the loss function is always fixed once you run `model.compile()`, and can't be changed in `model.fit()`."}]}}),{c(){h(c.$$.fragment),d=o(),h(f.$$.fragment)},l(n){p(c.$$.fragment,n),d=i(n),p(f.$$.fragment,n)},m(n,b){u(c,n,b),s(n,d,b),u(f,n,b),g=!0},i(n){g||(l(c.$$.fragment,n),l(f.$$.fragment,n),g=!0)},o(n){r(c.$$.fragment,n),r(f.$$.fragment,n),g=!1},d(n){n&&a(d),m(c,n),m(f,n)}}}function Re(k){let c,d,f,g;return c=new $({props:{title:"8. Why is there a specific subclass of Trainer for sequence-to-sequence problems?",local:"8-why-is-there-a-specific-subclass-of-trainer-for-sequence-to-sequence-problems",headingTag:"h3"}}),f=new w({props:{choices:[{text:"Because sequence-to-sequence problems use a custom loss, to ignore the labels set to <code>-100</code>",explain:"That's not a custom loss at all, but the way the loss is always computed."},{text:"Because sequence-to-sequence problems require a special evaluation loop",explain:"That's correct. Sequence-to-sequence models' predictions are often run using the <code>generate()</code> method.",correct:!0},{text:"Because the targets are texts in sequence-to-sequence problems",explain:"The <code>Trainer</code> doesn't really care about that since they have been preprocessed before."},{text:"Because we use two models in sequence-to-sequence problems",explain:"We do use two models in a way, an encoder and a decoder, but they are grouped together in one model."}]}}),{c(){h(c.$$.fragment),d=o(),h(f.$$.fragment)},l(n){p(c.$$.fragment,n),d=i(n),p(f.$$.fragment,n)},m(n,b){u(c,n,b),s(n,d,b),u(f,n,b),g=!0},i(n){g||(l(c.$$.fragment,n),l(f.$$.fragment,n),g=!0)},o(n){r(c.$$.fragment,n),r(f.$$.fragment,n),g=!1},d(n){n&&a(d),m(c,n),m(f,n)}}}function Ue(k){let c,d,f,g,n,b,q,V,T,X,v,We="Let’s test what you learned in this chapter!",Z,W,ee,_,te,z,ne,C,ae,B,se,N,oe,j,ie,E,re,F,le,S,he,A,pe,I,ue,P,me,H,ce,y,x,K,Y,fe,L,de,M,ge,Q,$e,R,we,U,be,D,ye,G,xe,O,ke;n=new Me({props:{fw:k[0]}}),q=new $({props:{title:"End-of-chapter quiz",local:"end-of-chapter-quiz",headingTag:"h1"}}),T=new Le({props:{chapter:7,classNames:"absolute z-10 right-0 top-0"}}),W=new $({props:{title:"1. Which of the following tasks can be framed as a token classification problem?",local:"1-which-of-the-following-tasks-can-be-framed-as-a-token-classification-problem",headingTag:"h3"}}),_=new w({props:{choices:[{text:"Find the grammatical components in a sentence.",explain:"Correct! We can then label each word as a noun, verb, etc.",correct:!0},{text:"Find whether a sentence is grammatically correct or not.",explain:"No, this is a sequence classification problem."},{text:"Find the persons mentioned in a sentence.",explain:"Correct! We can label each word as person or not person.",correct:!0},{text:"Find the chunk of words in a sentence that answers a question.",explain:"No, that would be a question answering problem."}]}}),z=new $({props:{title:"2. What part of the preprocessing for token classification differs from the other preprocessing pipelines?",local:"2-what-part-of-the-preprocessing-for-token-classification-differs-from-the-other-preprocessing-pipelines",headingTag:"h3"}}),C=new w({props:{choices:[{text:"There is no need to do anything; the texts are already tokenized.",explain:"The texts are indeed given as separate words, but we still need to apply the subword tokenization model."},{text:"The texts are given as words, so we only need to apply subword tokenization.",explain:"Correct! This is different from the usual preprocessing, where we need to apply the full tokenization pipeline. Can you think of another difference?",correct:!0},{text:"We use <code>-100</code> to label the special tokens.",explain:"That's not specific to token classification -- we always use <code>-100</code> as the label for tokens we want to ignore in the loss."},{text:"We need to make sure to truncate or pad the labels to the same size as the inputs, when applying truncation/padding.",explain:"Indeed! That's not the only difference, though.",correct:!0}]}}),B=new $({props:{title:"3. What problem arises when we tokenize the words in a token classification problem and want to label the tokens?",local:"3-what-problem-arises-when-we-tokenize-the-words-in-a-token-classification-problem-and-want-to-label-the-tokens",headingTag:"h3"}}),N=new w({props:{choices:[{text:"The tokenizer adds special tokens and we have no labels for them.",explain:"We label these <code>-100</code> so they are ignored in the loss."},{text:"Each word can produce several tokens, so we end up with more tokens than we have labels.",explain:"That is the main problem, and we need to align the original labels with the tokens.",correct:!0},{text:"The added tokens have no labels, so there is no problem.",explain:"That's incorrect; we need as many labels as we have tokens or our models will error out."}]}}),j=new $({props:{title:"4. What does “domain adaptation” mean?",local:"4-what-does-domain-adaptation-mean",headingTag:"h3"}}),E=new w({props:{choices:[{text:"It's when we run a model on a dataset and get the predictions for each sample in that dataset.",explain:"No, this is just running inference."},{text:"It's when we train a model on a dataset.",explain:"No, this is training a model; there is no adaptation here."},{text:"It's when we fine-tune a pretrained model on a new dataset, and it gives predictions that are more adapted to that dataset",explain:"Correct! The model adapted its knowledge to the new dataset.",correct:!0},{text:"It's when we add misclassified samples to a dataset to make our model more robust.",explain:"That's certainly something you should do if you retrain your model regularly, but it's not domain adaptation."}]}}),F=new $({props:{title:"5. What are the labels in a masked language modeling problem?",local:"5-what-are-the-labels-in-a-masked-language-modeling-problem",headingTag:"h3"}}),S=new w({props:{choices:[{text:"Some of the tokens in the input sentence are randomly masked and the labels are the original input tokens.",explain:"That's it!",correct:!0},{text:"Some of the tokens in the input sentence are randomly masked and the labels are the original input tokens, shifted to the left.",explain:"No, shifting the labels to the left corresponds to predicting the next word, which is causal language modeling."},{text:"Some of the tokens in the input sentence are randomly masked, and the label is whether the sentence is positive or negative.",explain:"That's a sequence classification problem with some data augmentation, not masked language modeling."},{text:"Some of the tokens in the two input sentences are randomly masked, and the label is whether the two sentences are similar or not.",explain:"That's a sequence classification problem with some data augmentation, not masked language modeling."}]}}),A=new $({props:{title:"6. Which of these tasks can be seen as a sequence-to-sequence problem?",local:"6-which-of-these-tasks-can-be-seen-as-a-sequence-to-sequence-problem",headingTag:"h3"}}),I=new w({props:{choices:[{text:"Writing short reviews of long documents",explain:"Yes, that's a summarization problem. Try another answer!",correct:!0},{text:"Answering questions about a document",explain:"This can be framed as a sequence-to-sequence problem. It's not the only right answer, though.",correct:!0},{text:"Translating a text in Chinese into English",explain:"That's definitely a sequence-to-sequence problem. Can you spot another one?",correct:!0},{text:"Fixing the messages sent by my nephew/friend so they're in proper English",explain:"That's a kind of translation problem, so definitely a sequence-to-sequence task. This isn't the only right answer, though!",correct:!0}]}}),P=new $({props:{title:"7. What is the proper way to preprocess the data for a sequence-to-sequence problem?",local:"7-what-is-the-proper-way-to-preprocess-the-data-for-a-sequence-to-sequence-problem",headingTag:"h3"}}),H=new w({props:{choices:[{text:"The inputs and targets have to be sent together to the tokenizer with <code>inputs=...</code> and <code>targets=...</code>.",explain:"This might be an API we add in the future, but that's not possible right now."},{text:"The inputs and the targets both have to be preprocessed, in two separate calls to the tokenizer.",explain:"That is true, but incomplete. There is something you need to do to make sure the tokenizer processes both properly."},{text:"As usual, we just have to tokenize the inputs.",explain:"Not in a sequence classification problem; the targets are also texts we need to convert into numbers!"},{text:"The inputs have to be sent to the tokenizer, and the targets too, but under a special context manager.",explain:"That's correct, the tokenizer needs to be put into target mode by that context manager.",correct:!0}]}});const _e=[Re,Qe],J=[];function ze(e,t){return e[0]==="pt"?0:1}return y=ze(k),x=J[y]=_e[y](k),Y=new $({props:{title:"10. When should you pretrain a new model?",local:"10-when-should-you-pretrain-a-new-model",headingTag:"h3"}}),L=new w({props:{choices:[{text:"When there is no pretrained model available for your specific language",explain:"That's correct.",correct:!0},{text:"When you have lots of data available, even if there is a pretrained model that could work on it",explain:"In this case, you should probably use the pretrained model and fine-tune it on your data, to avoid huge compute costs."},{text:"When you have concerns about the bias of the pretrained model you are using",explain:"That is true, but you have to make very sure the data you will use for training is really better.",correct:!0},{text:"When the pretrained models available are just not good enough",explain:"Are you sure you've properly debugged your training, then?"}]}}),M=new $({props:{title:"11. Why is it easy to pretrain a language model on lots and lots of texts?",local:"11-why-is-it-easy-to-pretrain-a-language-model-on-lots-and-lots-of-texts",headingTag:"h3"}}),Q=new w({props:{choices:[{text:"Because there are plenty of texts available on the internet",explain:"Although true, that doesn't really answer the question. Try again!"},{text:"Because the pretraining objective does not require humans to label the data",explain:"That's correct, language modeling is a self-supervised problem.",correct:!0},{text:"Because the 🤗 Transformers library only requires a few lines of code to start the training",explain:"Although true, that doesn't really answer the question asked. Try another answer!"}]}}),R=new $({props:{title:"12. What are the main challenges when preprocessing data for a question answering task?",local:"12-what-are-the-main-challenges-when-preprocessing-data-for-a-question-answering-task",headingTag:"h3"}}),U=new w({props:{choices:[{text:"You need to tokenize the inputs.",explain:"That's correct, but is it really a main challenge?"},{text:"You need to deal with very long contexts, which give several training features that may or may not have the answer in them.",explain:"This is definitely one of the challenges.",correct:!0},{text:"You need to tokenize the answers to the question as well as the inputs.",explain:"No, unless you are framing your question answering problem as a sequence-to-sequence task."},{text:"From the answer span in the text, you have to find the start and end token in the tokenized input.",explain:"That's one of the hard parts, yes!",correct:!0}]}}),D=new $({props:{title:"13. How is post-processing usually done in question answering?",local:"13-how-is-post-processing-usually-done-in-question-answering",headingTag:"h3"}}),G=new w({props:{choices:[{text:"The model gives you the start and end positions of the answer, and you just have to decode the corresponding span of tokens.",explain:"That could be one way to do it, but it's a bit too simplistic."},{text:"The model gives you the start and end positions of the answer for each feature created by one example, and you just have to decode the corresponding span of tokens in the one that has the best score.",explain:"That's close to the post-processing we studied, but it's not entirely right."},{text:"The model gives you the start and end positions of the answer for each feature created by one example, and you just have to match them to the span in the context for the one that has the best score.",explain:"That's it in a nutshell!",correct:!0},{text:"The model generates an answer, and you just have to decode it.",explain:"No, unless you are framing your question answering problem as a sequence-to-sequence task."}]}}),{c(){c=qe("meta"),d=o(),f=qe("p"),g=o(),h(n.$$.fragment),b=o(),h(q.$$.fragment),V=o(),h(T.$$.fragment),X=o(),v=qe("p"),v.textContent=We,Z=o(),h(W.$$.fragment),ee=o(),h(_.$$.fragment),te=o(),h(z.$$.fragment),ne=o(),h(C.$$.fragment),ae=o(),h(B.$$.fragment),se=o(),h(N.$$.fragment),oe=o(),h(j.$$.fragment),ie=o(),h(E.$$.fragment),re=o(),h(F.$$.fragment),le=o(),h(S.$$.fragment),he=o(),h(A.$$.fragment),pe=o(),h(I.$$.fragment),ue=o(),h(P.$$.fragment),me=o(),h(H.$$.fragment),ce=o(),x.c(),K=o(),h(Y.$$.fragment),fe=o(),h(L.$$.fragment),de=o(),h(M.$$.fragment),ge=o(),h(Q.$$.fragment),$e=o(),h(R.$$.fragment),we=o(),h(U.$$.fragment),be=o(),h(D.$$.fragment),ye=o(),h(G.$$.fragment),xe=o(),O=qe("p"),this.h()},l(e){const t=Ae("svelte-u9bgzb",document.head);c=Te(t,"META",{name:!0,content:!0}),t.forEach(a),d=i(e),f=Te(e,"P",{}),Be(f).forEach(a),g=i(e),p(n.$$.fragment,e),b=i(e),p(q.$$.fragment,e),V=i(e),p(T.$$.fragment,e),X=i(e),v=Te(e,"P",{"data-svelte-h":!0}),Ie(v)!=="svelte-19og2hy"&&(v.textContent=We),Z=i(e),p(W.$$.fragment,e),ee=i(e),p(_.$$.fragment,e),te=i(e),p(z.$$.fragment,e),ne=i(e),p(C.$$.fragment,e),ae=i(e),p(B.$$.fragment,e),se=i(e),p(N.$$.fragment,e),oe=i(e),p(j.$$.fragment,e),ie=i(e),p(E.$$.fragment,e),re=i(e),p(F.$$.fragment,e),le=i(e),p(S.$$.fragment,e),he=i(e),p(A.$$.fragment,e),pe=i(e),p(I.$$.fragment,e),ue=i(e),p(P.$$.fragment,e),me=i(e),p(H.$$.fragment,e),ce=i(e),x.l(e),K=i(e),p(Y.$$.fragment,e),fe=i(e),p(L.$$.fragment,e),de=i(e),p(M.$$.fragment,e),ge=i(e),p(Q.$$.fragment,e),$e=i(e),p(R.$$.fragment,e),we=i(e),p(U.$$.fragment,e),be=i(e),p(D.$$.fragment,e),ye=i(e),p(G.$$.fragment,e),xe=i(e),O=Te(e,"P",{}),Be(O).forEach(a),this.h()},h(){Ne(c,"name","hf:doc:metadata"),Ne(c,"content",De)},m(e,t){Pe(document.head,c),s(e,d,t),s(e,f,t),s(e,g,t),u(n,e,t),s(e,b,t),u(q,e,t),s(e,V,t),u(T,e,t),s(e,X,t),s(e,v,t),s(e,Z,t),u(W,e,t),s(e,ee,t),u(_,e,t),s(e,te,t),u(z,e,t),s(e,ne,t),u(C,e,t),s(e,ae,t),u(B,e,t),s(e,se,t),u(N,e,t),s(e,oe,t),u(j,e,t),s(e,ie,t),u(E,e,t),s(e,re,t),u(F,e,t),s(e,le,t),u(S,e,t),s(e,he,t),u(A,e,t),s(e,pe,t),u(I,e,t),s(e,ue,t),u(P,e,t),s(e,me,t),u(H,e,t),s(e,ce,t),J[y].m(e,t),s(e,K,t),u(Y,e,t),s(e,fe,t),u(L,e,t),s(e,de,t),u(M,e,t),s(e,ge,t),u(Q,e,t),s(e,$e,t),u(R,e,t),s(e,we,t),u(U,e,t),s(e,be,t),u(D,e,t),s(e,ye,t),u(G,e,t),s(e,xe,t),s(e,O,t),ke=!0},p(e,[t]){const Ce={};t&1&&(Ce.fw=e[0]),n.$set(Ce);let ve=y;y=ze(e),y!==ve&&(Ye(),r(J[ve],1,1,()=>{J[ve]=null}),He(),x=J[y],x||(x=J[y]=_e[y](e),x.c()),l(x,1),x.m(K.parentNode,K))},i(e){ke||(l(n.$$.fragment,e),l(q.$$.fragment,e),l(T.$$.fragment,e),l(W.$$.fragment,e),l(_.$$.fragment,e),l(z.$$.fragment,e),l(C.$$.fragment,e),l(B.$$.fragment,e),l(N.$$.fragment,e),l(j.$$.fragment,e),l(E.$$.fragment,e),l(F.$$.fragment,e),l(S.$$.fragment,e),l(A.$$.fragment,e),l(I.$$.fragment,e),l(P.$$.fragment,e),l(H.$$.fragment,e),l(x),l(Y.$$.fragment,e),l(L.$$.fragment,e),l(M.$$.fragment,e),l(Q.$$.fragment,e),l(R.$$.fragment,e),l(U.$$.fragment,e),l(D.$$.fragment,e),l(G.$$.fragment,e),ke=!0)},o(e){r(n.$$.fragment,e),r(q.$$.fragment,e),r(T.$$.fragment,e),r(W.$$.fragment,e),r(_.$$.fragment,e),r(z.$$.fragment,e),r(C.$$.fragment,e),r(B.$$.fragment,e),r(N.$$.fragment,e),r(j.$$.fragment,e),r(E.$$.fragment,e),r(F.$$.fragment,e),r(S.$$.fragment,e),r(A.$$.fragment,e),r(I.$$.fragment,e),r(P.$$.fragment,e),r(H.$$.fragment,e),r(x),r(Y.$$.fragment,e),r(L.$$.fragment,e),r(M.$$.fragment,e),r(Q.$$.fragment,e),r(R.$$.fragment,e),r(U.$$.fragment,e),r(D.$$.fragment,e),r(G.$$.fragment,e),ke=!1},d(e){e&&(a(d),a(f),a(g),a(b),a(V),a(X),a(v),a(Z),a(ee),a(te),a(ne),a(ae),a(se),a(oe),a(ie),a(re),a(le),a(he),a(pe),a(ue),a(me),a(ce),a(K),a(fe),a(de),a(ge),a($e),a(we),a(be),a(ye),a(xe),a(O)),a(c),m(n,e),m(q,e),m(T,e),m(W,e),m(_,e),m(z,e),m(C,e),m(B,e),m(N,e),m(j,e),m(E,e),m(F,e),m(S,e),m(A,e),m(I,e),m(P,e),m(H,e),J[y].d(e),m(Y,e),m(L,e),m(M,e),m(Q,e),m(R,e),m(U,e),m(D,e),m(G,e)}}}const De='{"title":"End-of-chapter quiz","local":"end-of-chapter-quiz","sections":[{"title":"1. Which of the following tasks can be framed as a token classification problem?","local":"1-which-of-the-following-tasks-can-be-framed-as-a-token-classification-problem","sections":[],"depth":3},{"title":"2. What part of the preprocessing for token classification differs from the other preprocessing pipelines?","local":"2-what-part-of-the-preprocessing-for-token-classification-differs-from-the-other-preprocessing-pipelines","sections":[],"depth":3},{"title":"3. What problem arises when we tokenize the words in a token classification problem and want to label the tokens?","local":"3-what-problem-arises-when-we-tokenize-the-words-in-a-token-classification-problem-and-want-to-label-the-tokens","sections":[],"depth":3},{"title":"4. What does “domain adaptation” mean?","local":"4-what-does-domain-adaptation-mean","sections":[],"depth":3},{"title":"5. What are the labels in a masked language modeling problem?","local":"5-what-are-the-labels-in-a-masked-language-modeling-problem","sections":[],"depth":3},{"title":"6. Which of these tasks can be seen as a sequence-to-sequence problem?","local":"6-which-of-these-tasks-can-be-seen-as-a-sequence-to-sequence-problem","sections":[],"depth":3},{"title":"7. What is the proper way to preprocess the data for a sequence-to-sequence problem?","local":"7-what-is-the-proper-way-to-preprocess-the-data-for-a-sequence-to-sequence-problem","sections":[],"depth":3},{"title":"8. Why is there a specific subclass of Trainer for sequence-to-sequence problems?","local":"8-why-is-there-a-specific-subclass-of-trainer-for-sequence-to-sequence-problems","sections":[],"depth":3},{"title":"9. Why is it often unnecessary to specify a loss when calling compile() on a Transformer model?","local":"9-why-is-it-often-unnecessary-to-specify-a-loss-when-calling-compile-on-a-transformer-model","sections":[],"depth":3},{"title":"10. When should you pretrain a new model?","local":"10-when-should-you-pretrain-a-new-model","sections":[],"depth":3},{"title":"11. Why is it easy to pretrain a language model on lots and lots of texts?","local":"11-why-is-it-easy-to-pretrain-a-language-model-on-lots-and-lots-of-texts","sections":[],"depth":3},{"title":"12. What are the main challenges when preprocessing data for a question answering task?","local":"12-what-are-the-main-challenges-when-preprocessing-data-for-a-question-answering-task","sections":[],"depth":3},{"title":"13. How is post-processing usually done in question answering?","local":"13-how-is-post-processing-usually-done-in-question-answering","sections":[],"depth":3}],"depth":1}';function Ge(k,c,d){let f="pt";return Ee(()=>{const g=new URLSearchParams(window.location.search);d(0,f=g.get("fw")||"pt")}),[f]}class et extends Fe{constructor(c){super(),Se(this,c,Ge,Ue,je,{})}}export{et as component};
