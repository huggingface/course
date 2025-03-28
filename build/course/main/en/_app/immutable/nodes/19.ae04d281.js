import{s as fe,n as pe,o as de}from"../chunks/scheduler.37c15a92.js";import{S as ge,i as he,g as ne,s as n,r as i,A as ce,h as oe,f as a,c as o,j as le,u as r,x as me,k as ue,y as $e,a as s,v as l,d as u,t as f,w as p}from"../chunks/index.2bf4358c.js";import{C as ye}from"../chunks/CourseFloatingBanner.15ba07e6.js";import{Q as h}from"../chunks/Question.668688bc.js";import{H as g}from"../chunks/Heading.8ada512a.js";function xe(ie){let d,C,L,N,c,Q,m,z,$,re="Let’s test what you learned in this chapter!",E,y,K,x,I,w,M,b,j,k,O,A,U,v,B,T,G,_,R,q,J,S,V,Y,X,H,Z,F,ee,D,te,P,ae,W,se;return c=new g({props:{title:"End-of-chapter quiz",local:"end-of-chapter-quiz",headingTag:"h1"}}),m=new ye({props:{chapter:10,classNames:"absolute z-10 right-0 top-0"}}),y=new g({props:{title:"1. What can you use Argilla for?",local:"1-what-can-you-use-argilla-for",headingTag:"h3"}}),x=new h({props:{choices:[{text:"Turn unstructured data into structured data for NLP tasks",explain:"You can use Argilla to add annotations to a dataset and use it for NLP tasks.",correct:!0},{text:"Scrap a public website to build a dataset",explain:"This is not a feature in Argilla, but you can scrap a public website and turn it into an Argilla dataset for annotations using the Python SDK."},{text:"Improve the quality of an existing dataset",explain:"You can use previous annotations as suggestions and correct them to improve the quality of a dataset.",correct:!0},{text:"Adapt an existing dataset to your own use case",explain:"You can use different question types in Argilla to adapt an existing dataset to your own use case.",correct:!0},{text:"Train your model",explain:"You cannot train a model directly in Argilla, but you can use the data you curate in Argilla to train your own model"},{text:"Generate synthetic datasets",explain:"To generate synthetic datasets, you can use the distilabel package and then use Argilla to review and curate the generated data."}]}}),w=new g({props:{title:"2. Argilla ONLY works in the Hugging Face Spaces and with Hugging Face Datasets.",local:"2-argilla-only-works-in-the-hugging-face-spaces-and-with-hugging-face-datasets",headingTag:"h3"}}),b=new h({props:{choices:[{text:"True",explain:"You can also deploy Argilla locally using Docker and you can use the Python SDK to upload any type of data, including Hugging Face Datasets."},{text:"False",explain:"You can also deploy Argilla locally using Docker and you can use the Python SDK to upload any type of data, including Hugging Face Datasets.",correct:!0}]}}),k=new g({props:{title:"3. You need a Hugging Face token to connect the Python SDK to your Argilla server.",local:"3-you-need-a-hugging-face-token-to-connect-the-python-sdk-to-your-argilla-server",headingTag:"h3"}}),A=new h({props:{choices:[{text:"True",explain:"This is only needed if your Argilla Space is private!"},{text:"False",explain:"You don't need a token if you are using a public Argilla Space or a local deployment with Docker.",correct:!0}]}}),v=new g({props:{title:"4. What are fields in Argilla? How many fields can you use?",local:"4-what-are-fields-in-argilla-how-many-fields-can-you-use",headingTag:"h3"}}),T=new h({props:{choices:[{text:"Fields show the data that we are annotating. All this information needs to be collected in a single field.",explain:"You can spread the information across multiple fields, if you need to."},{text:"Fields show the data that we are annotating. All this information can be spread across multiple fields.",explain:"Yes, you can have multiple fields and also fields of different types (text, chat, image...) depending on the type of data you're annotating.",correct:!0},{text:"Fields contain the metadata of the records. You can use as many as you need.",explain:"You can have both fields and metadata in your dataset, but they serve separate purposes. Metadata are used for filtering and sorting purposes as extra information, while fields show the data that we are annotating."}]}}),_=new g({props:{title:"5. What’s the best type of question for a token classification task?",local:"5-whats-the-best-type-of-question-for-a-token-classification-task",headingTag:"h3"}}),q=new h({props:{choices:[{text:"A SpanQuestion",explain:"SpanQuestions let you highlight bits of text and apply a label to them. This is the best type for a token classification task.",correct:!0},{text:"A LabelQuestion",explain:"This type of question allows you to select a label that applies to the whole record. This type is best for a text classification task."},{text:"A TextQuestion",explain:"This type of question allows you to write text. This is not suitable for a token classfication task."},{text:"None of the above",explain:"SpanQuestions let you highlight bits of text and apply a label to them. This is the best type for a token classification task."}]}}),S=new g({props:{title:"6. What is the purpose of the “Save as draft” button?",local:"6-what-is-the-purpose-of-the-save-as-draft-button",headingTag:"h3"}}),Y=new h({props:{choices:[{text:"Submit your responses",explain:"This button saves your responses, but doesn't submit them",correct:!0},{text:"Save your responses without submitting them",explain:"This is a valid method of loading a Hugging Face model from the Hub",correct:!0},{text:"Discard a record",explain:"Try again -- you cannot load a model by using the 'demos' prefix."}]}}),H=new g({props:{title:"7. Argilla does not offer suggested labels automatically, you need to provide that data yourself.",local:"7-argilla-does-not-offer-suggested-labels-automatically-you-need-to-provide-that-data-yourself",headingTag:"h3"}}),F=new h({props:{choices:[{text:"True",explain:"You can add suggestions to your records (or update them) at any point of the project.",correct:!0},{text:"False",explain:"If you want to see suggested labels, you need to log them yourself when you create the records or at a later point."}]}}),D=new g({props:{title:"8. Select all the necessary steps to export an Argilla dataset in full to the Hub:",local:"8-select-all-the-necessary-steps-to-export-an-argilla-dataset-in-full-to-the-hub",headingTag:"h3"}}),P=new h({props:{choices:[{text:"You need to be connected to your Argilla server: <code>client= rg.Argilla(api_url='...', api_key='...')</code>",explain:"Yes, to interact with your server you'll need to instantiate it first.",correct:!0},{text:"Import the dataset from the hub: <code>dataset = rg.Dataset.from_hub(repo_id='argilla/ag_news_annotated')</code>",explain:"No. This is to import a dataset from the Hub into your Argilla instance."},{text:"Load the dataset: <code>dataset = client.datasets(name='my_dataset')</code>",explain:"Yes, you'll need this for further operations",correct:!0},{text:"Convert the Argilla dataset into a Datasets dataset: <code>dataset = dataset.to_datasets()</code>",explain:"This is not needed if you export the full dataset. Argilla will take care of this for you. However, you might need it if you're working with a subset of records."},{text:"Use the </code>to_hub</code> method to export the dataset: <code>dataset.to_hub(repo_id='my_username/dataset_name')</code>",explain:"This will push the dataset to the indicated repo id, and create a new repo if it doesn't exist.",correct:!0}]}}),{c(){d=ne("meta"),C=n(),L=ne("p"),N=n(),i(c.$$.fragment),Q=n(),i(m.$$.fragment),z=n(),$=ne("p"),$.textContent=re,E=n(),i(y.$$.fragment),K=n(),i(x.$$.fragment),I=n(),i(w.$$.fragment),M=n(),i(b.$$.fragment),j=n(),i(k.$$.fragment),O=n(),i(A.$$.fragment),U=n(),i(v.$$.fragment),B=n(),i(T.$$.fragment),G=n(),i(_.$$.fragment),R=n(),i(q.$$.fragment),J=n(),i(S.$$.fragment),V=n(),i(Y.$$.fragment),X=n(),i(H.$$.fragment),Z=n(),i(F.$$.fragment),ee=n(),i(D.$$.fragment),te=n(),i(P.$$.fragment),ae=n(),W=ne("p"),this.h()},l(e){const t=ce("svelte-u9bgzb",document.head);d=oe(t,"META",{name:!0,content:!0}),t.forEach(a),C=o(e),L=oe(e,"P",{}),le(L).forEach(a),N=o(e),r(c.$$.fragment,e),Q=o(e),r(m.$$.fragment,e),z=o(e),$=oe(e,"P",{"data-svelte-h":!0}),me($)!=="svelte-19og2hy"&&($.textContent=re),E=o(e),r(y.$$.fragment,e),K=o(e),r(x.$$.fragment,e),I=o(e),r(w.$$.fragment,e),M=o(e),r(b.$$.fragment,e),j=o(e),r(k.$$.fragment,e),O=o(e),r(A.$$.fragment,e),U=o(e),r(v.$$.fragment,e),B=o(e),r(T.$$.fragment,e),G=o(e),r(_.$$.fragment,e),R=o(e),r(q.$$.fragment,e),J=o(e),r(S.$$.fragment,e),V=o(e),r(Y.$$.fragment,e),X=o(e),r(H.$$.fragment,e),Z=o(e),r(F.$$.fragment,e),ee=o(e),r(D.$$.fragment,e),te=o(e),r(P.$$.fragment,e),ae=o(e),W=oe(e,"P",{}),le(W).forEach(a),this.h()},h(){ue(d,"name","hf:doc:metadata"),ue(d,"content",we)},m(e,t){$e(document.head,d),s(e,C,t),s(e,L,t),s(e,N,t),l(c,e,t),s(e,Q,t),l(m,e,t),s(e,z,t),s(e,$,t),s(e,E,t),l(y,e,t),s(e,K,t),l(x,e,t),s(e,I,t),l(w,e,t),s(e,M,t),l(b,e,t),s(e,j,t),l(k,e,t),s(e,O,t),l(A,e,t),s(e,U,t),l(v,e,t),s(e,B,t),l(T,e,t),s(e,G,t),l(_,e,t),s(e,R,t),l(q,e,t),s(e,J,t),l(S,e,t),s(e,V,t),l(Y,e,t),s(e,X,t),l(H,e,t),s(e,Z,t),l(F,e,t),s(e,ee,t),l(D,e,t),s(e,te,t),l(P,e,t),s(e,ae,t),s(e,W,t),se=!0},p:pe,i(e){se||(u(c.$$.fragment,e),u(m.$$.fragment,e),u(y.$$.fragment,e),u(x.$$.fragment,e),u(w.$$.fragment,e),u(b.$$.fragment,e),u(k.$$.fragment,e),u(A.$$.fragment,e),u(v.$$.fragment,e),u(T.$$.fragment,e),u(_.$$.fragment,e),u(q.$$.fragment,e),u(S.$$.fragment,e),u(Y.$$.fragment,e),u(H.$$.fragment,e),u(F.$$.fragment,e),u(D.$$.fragment,e),u(P.$$.fragment,e),se=!0)},o(e){f(c.$$.fragment,e),f(m.$$.fragment,e),f(y.$$.fragment,e),f(x.$$.fragment,e),f(w.$$.fragment,e),f(b.$$.fragment,e),f(k.$$.fragment,e),f(A.$$.fragment,e),f(v.$$.fragment,e),f(T.$$.fragment,e),f(_.$$.fragment,e),f(q.$$.fragment,e),f(S.$$.fragment,e),f(Y.$$.fragment,e),f(H.$$.fragment,e),f(F.$$.fragment,e),f(D.$$.fragment,e),f(P.$$.fragment,e),se=!1},d(e){e&&(a(C),a(L),a(N),a(Q),a(z),a($),a(E),a(K),a(I),a(M),a(j),a(O),a(U),a(B),a(G),a(R),a(J),a(V),a(X),a(Z),a(ee),a(te),a(ae),a(W)),a(d),p(c,e),p(m,e),p(y,e),p(x,e),p(w,e),p(b,e),p(k,e),p(A,e),p(v,e),p(T,e),p(_,e),p(q,e),p(S,e),p(Y,e),p(H,e),p(F,e),p(D,e),p(P,e)}}}const we='{"title":"End-of-chapter quiz","local":"end-of-chapter-quiz","sections":[{"title":"1. What can you use Argilla for?","local":"1-what-can-you-use-argilla-for","sections":[],"depth":3},{"title":"2. Argilla ONLY works in the Hugging Face Spaces and with Hugging Face Datasets.","local":"2-argilla-only-works-in-the-hugging-face-spaces-and-with-hugging-face-datasets","sections":[],"depth":3},{"title":"3. You need a Hugging Face token to connect the Python SDK to your Argilla server.","local":"3-you-need-a-hugging-face-token-to-connect-the-python-sdk-to-your-argilla-server","sections":[],"depth":3},{"title":"4. What are fields in Argilla? How many fields can you use?","local":"4-what-are-fields-in-argilla-how-many-fields-can-you-use","sections":[],"depth":3},{"title":"5. What’s the best type of question for a token classification task?","local":"5-whats-the-best-type-of-question-for-a-token-classification-task","sections":[],"depth":3},{"title":"6. What is the purpose of the “Save as draft” button?","local":"6-what-is-the-purpose-of-the-save-as-draft-button","sections":[],"depth":3},{"title":"7. Argilla does not offer suggested labels automatically, you need to provide that data yourself.","local":"7-argilla-does-not-offer-suggested-labels-automatically-you-need-to-provide-that-data-yourself","sections":[],"depth":3},{"title":"8. Select all the necessary steps to export an Argilla dataset in full to the Hub:","local":"8-select-all-the-necessary-steps-to-export-an-argilla-dataset-in-full-to-the-hub","sections":[],"depth":3}],"depth":1}';function be(ie){return de(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class qe extends ge{constructor(d){super(),he(this,d,be,xe,fe,{})}}export{qe as component};
