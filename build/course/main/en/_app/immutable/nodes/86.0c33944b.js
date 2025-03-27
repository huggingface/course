import{s as Nt,n as zt,o as Qt}from"../chunks/scheduler.37c15a92.js";import{S as Lt,i as Et,g as n,s as o,r as h,A as _t,h as s,f as l,c as a,j as Xt,u as c,x as r,k as y,y as Pt,a as i,v as m,d as u,t as d,w as p}from"../chunks/index.2bf4358c.js";import{Y as qt}from"../chunks/Youtube.1e50a667.js";import{C as at}from"../chunks/CodeBlock.4f5fc1ad.js";import{C as At}from"../chunks/CourseFloatingBanner.15ba07e6.js";import{H as re}from"../chunks/Heading.8ada512a.js";function Dt(nt){let M,me,he,ue,U,de,j,pe,G,Me,B,st='The <a href="https://discuss.huggingface.co" rel="nofollow">Hugging Face forums</a> are a great place to get help from the open source team and wider Hugging Face community. Here’s what the main page looks like on any given day:',ye,w,rt='<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter8/forums.png" alt="The Hugging Face forums." width="100%"/>',we,I,ht='On the lefthand side you can see all the categories that the various topics are grouped into, while the righthand side shows the most recent topics. A topic is a post that contains a title, category, and description; it’s quite similar to the GitHub issues format that we saw when creating our own dataset in <a href="/course/chapter5">Chapter 5</a>. As the name suggests, the <a href="https://discuss.huggingface.co/c/beginners/5" rel="nofollow">Beginners</a> category is primarily intended for people just starting out with the Hugging Face libraries and ecosystem. Any question on any of the libraries is welcome there, be it to debug some code or to ask for help about how to do something. (That said, if your question concerns one library in particular, you should probably head to the corresponding library category on the forum.)',fe,k,ct='Similarly, the <a href="https://discuss.huggingface.co/c/intermediate/6" rel="nofollow">Intermediate</a> and <a href="https://discuss.huggingface.co/c/research/7" rel="nofollow">Research</a> categories are for more advanced questions, for example about the libraries or some cool new NLP research that you’d like to discuss.',be,Z,mt='And naturally, we should also mention the <a href="https://discuss.huggingface.co/c/course/20" rel="nofollow">Course</a> category, where you can ask any questions you have that are related to the Hugging Face course!',ge,W,ut='Once you have selected a category, you’ll be ready to write your first topic. You can find some <a href="https://discuss.huggingface.co/t/how-to-request-support/3128" rel="nofollow">guidelines</a> in the forum on how to do this, and in this section we’ll take a look at some features that make up a good topic.',Te,H,Je,x,dt="As a running example, let’s suppose that we’re trying to generate embeddings from Wikipedia articles to create a custom search engine. As usual, we load the tokenizer and model as follows:",ve,V,Ue,C,pt='Now suppose we try to embed a whole section of the <a href="https://en.wikipedia.org/wiki/Transformers" rel="nofollow">Wikipedia article</a> on Transformers (the franchise, not the library!):',je,Y,Ge,F,Be,S,Mt='Uh-oh, we’ve hit a problem — and the error message is far more cryptic than the ones we saw in <a href="/course/chapter8/section2">section 2</a>! We can’t make head or tails of the full traceback, so we decide to turn to the Hugging Face forums for help. How might we craft the topic?',Ie,$,yt="To get started, we need to click the “New Topic” button at the upper-right corner (note that to create a topic, we’ll need to be logged in):",ke,f,wt='<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter8/forums-new-topic.png" alt="Creating a new forum topic." width="100%"/>',Ze,R,ft="This brings up a writing interface where we can input the title of our topic, select a category, and draft the content:",We,b,bt='<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter8/forum-topic01.png" alt="The interface for creating a forum topic." width="100%"/>',He,X,gt="Since the error seems to be exclusively about 🤗 Transformers, we’ll select this for the category. Our first attempt at explaining the problem might look something like this:",xe,g,Tt='<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter8/forum-topic02.png" alt="Drafting the content for a new forum topic." width="100%"/>',Ve,N,Jt="Although this topic contains the error message we need help with, there are a few problems with the way it is written:",Ce,z,vt="<li>The title is not very descriptive, so anyone browsing the forum won’t be able to tell what the topic is about without reading the body as well.</li> <li>The body doesn’t provide enough information about <em>where</em> the error is coming from and <em>how</em> to reproduce it.</li> <li>The topic tags a few people directly with a somewhat demanding tone.</li>",Ye,Q,Ut="Topics like this one are not likely to get a fast answer (if they get one at all), so let’s look at how we can improve it. We’ll start with the first issue of picking a good title.",Fe,L,Se,E,jt="If you’re trying to get help with a bug in your code, a good rule of thumb is to include enough information in the title so that others can quickly determine whether they think they can answer your question or not. In our running example, we know the name of the exception that’s being raised and have some hints that it’s triggered in the forward pass of the model, where we call <code>model(**inputs)</code>. To communicate this, one possible title could be:",$e,_,Gt="<p>Source of IndexError in the AutoModel forward pass?</p>",Re,P,Bt="This title tells the reader <em>where</em> you think the bug is coming from, and if they’ve encountered an <code>IndexError</code> before, there’s a good chance they’ll know how to debug it. Of course, the title can be anything you want, and other variations like:",Xe,q,It="<p>Why does my model produce an IndexError?</p>",Ne,A,kt="could also be fine. Now that we’ve got a descriptive title, let’s take a look at improving the body.",ze,D,Qe,O,Zt="Reading source code is hard enough in an IDE, but it’s even harder when the code is copied and pasted as plain text! Fortunately, the Hugging Face forums support the use of Markdown, so you should always enclose your code blocks with three backticks (```) so it’s more easily readable. Let’s do this to prettify the error message — and while we’re at it, let’s make the body a bit more polite than our original version:",Le,T,Wt='<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter8/forum-topic03.png" alt="Our revised forum topic, with proper code formatting." width="100%"/>',Ee,K,Ht="As you can see in the screenshot, enclosing the code blocks in backticks converts the raw text into formatted code, complete with color styling! Also note that single backticks can be used to format inline variables, like we’ve done for <code>distilbert-base-uncased</code>. This topic is looking much better, and with a bit of luck we might find someone in the community who can guess what the error is about. However, instead of relying on luck, let’s make life easier by including the traceback in its full gory detail!",_e,ee,Pe,te,xt="Since the last line of the traceback is often enough to debug your own code, it can be tempting to just provide that in your topic to “save space.” Although well intentioned, this actually makes it <em>harder</em> for others to debug the problem since the information that’s higher up in the traceback can be really useful too. So, a good practice is to copy and paste the <em>whole</em> traceback, while making sure that it’s nicely formatted. Since these tracebacks can get rather long, some people prefer to show them after they’ve explained the source code. Let’s do this. Now, our forum topic looks like the following:",qe,J,Vt='<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter8/forum-topic04.png" alt="Our example forum topic, with the complete traceback." width="100%"/>',Ae,le,Ct="This is much more informative, and a careful reader might be able to point out that the problem seems to be due to passing a long input because of this line in the traceback:",De,ie,Yt="<p>Token indices sequence length is longer than the specified maximum sequence length for this model (583 &gt; 512).</p>",Oe,oe,Ft="However, we can make things even easier for them by providing the actual code that triggered the error. Let’s do that now.",Ke,ae,et,ne,St="If you’ve ever tried to debug someone else’s code, you’ve probably first tried to recreate the problem they’ve reported so you can start working your way through the traceback to pinpoint the error. It’s no different when it comes to getting (or giving) assistance on the forums, so it really helps if you can provide a small example that reproduces the error. Half the time, simply walking through this exercise will help you figure out what’s going wrong. In any case, the missing piece of our example is to show the <em>inputs</em> that we provided to the model. Doing that gives us something like the following completed example:",tt,v,$t='<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter8/forum-topic05.png" alt="The final version of our forum topic." width="100%"/>',lt,se,Rt="This topic now contains quite a lot of information, and it’s written in a way that is much more likely to attract the attention of the community and get a helpful answer. With these basic guidelines, you can now create great topics to find the answers to your 🤗 Transformers questions!",it,ce,ot;return U=new re({props:{title:"Asking for help on the forums",local:"asking-for-help-on-the-forums",headingTag:"h1"}}),j=new At({props:{chapter:8,classNames:"absolute z-10 right-0 top-0",notebooks:[{label:"Google Colab",value:"https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter8/section3.ipynb"},{label:"Aws Studio",value:"https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter8/section3.ipynb"}]}}),G=new qt({props:{id:"S2EEG3JIt2A"}}),H=new re({props:{title:"Writing a good forum post",local:"writing-a-good-forum-post",headingTag:"h2"}}),V=new at({props:{code:"ZnJvbSUyMHRyYW5zZm9ybWVycyUyMGltcG9ydCUyMEF1dG9Ub2tlbml6ZXIlMkMlMjBBdXRvTW9kZWwlMEElMEFtb2RlbF9jaGVja3BvaW50JTIwJTNEJTIwJTIyZGlzdGlsYmVydC1iYXNlLXVuY2FzZWQlMjIlMEF0b2tlbml6ZXIlMjAlM0QlMjBBdXRvVG9rZW5pemVyLmZyb21fcHJldHJhaW5lZChtb2RlbF9jaGVja3BvaW50KSUwQW1vZGVsJTIwJTNEJTIwQXV0b01vZGVsLmZyb21fcHJldHJhaW5lZChtb2RlbF9jaGVja3BvaW50KQ==",highlighted:`<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoTokenizer, AutoModel

model_checkpoint = <span class="hljs-string">&quot;distilbert-base-uncased&quot;</span>
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModel.from_pretrained(model_checkpoint)`,wrap:!1}}),Y=new at({props:{code:"dGV4dCUyMCUzRCUyMCUyMiUyMiUyMiUwQUdlbmVyYXRpb24lMjBPbmUlMjBpcyUyMGElMjByZXRyb2FjdGl2ZSUyMHRlcm0lMjBmb3IlMjB0aGUlMjBUcmFuc2Zvcm1lcnMlMjBjaGFyYWN0ZXJzJTIwdGhhdCUwQWFwcGVhcmVkJTIwYmV0d2VlbiUyMDE5ODQlMjBhbmQlMjAxOTkzLiUyMFRoZSUyMFRyYW5zZm9ybWVycyUyMGJlZ2FuJTIwd2l0aCUyMHRoZSUyMDE5ODBzJTIwSmFwYW5lc2UlMEF0b3klMjBsaW5lcyUyME1pY3JvJTIwQ2hhbmdlJTIwYW5kJTIwRGlhY2xvbmUuJTIwVGhleSUyMHByZXNlbnRlZCUyMHJvYm90cyUyMGFibGUlMjB0byUyMHRyYW5zZm9ybSUwQWludG8lMjBldmVyeWRheSUyMHZlaGljbGVzJTJDJTIwZWxlY3Ryb25pYyUyMGl0ZW1zJTIwb3IlMjB3ZWFwb25zLiUyMEhhc2JybyUyMGJvdWdodCUyMHRoZSUyME1pY3JvJTBBQ2hhbmdlJTIwYW5kJTIwRGlhY2xvbmUlMjB0b3lzJTJDJTIwYW5kJTIwcGFydG5lcmVkJTIwd2l0aCUyMFRha2FyYS4lMjBNYXJ2ZWwlMjBDb21pY3MlMjB3YXMlMjBoaXJlZCUyMGJ5JTBBSGFzYnJvJTIwdG8lMjBjcmVhdGUlMjB0aGUlMjBiYWNrc3RvcnklM0IlMjBlZGl0b3ItaW4tY2hpZWYlMjBKaW0lMjBTaG9vdGVyJTIwd3JvdGUlMjBhbiUyMG92ZXJhbGwlMEFzdG9yeSUyQyUyMGFuZCUyMGdhdmUlMjB0aGUlMjB0YXNrJTIwb2YlMjBjcmVhdGluZyUyMHRoZSUyMGNoYXJhY3RoZXJzJTIwdG8lMjB3cml0ZXIlMjBEZW5uaXMlMjBPJ05laWwuJTBBVW5oYXBweSUyMHdpdGglMjBPJ05laWwncyUyMHdvcmslMjAoYWx0aG91Z2glMjBPJ05laWwlMjBjcmVhdGVkJTIwdGhlJTIwbmFtZSUyMCUyMk9wdGltdXMlMjBQcmltZSUyMiklMkMlMEFTaG9vdGVyJTIwY2hvc2UlMjBCb2IlMjBCdWRpYW5za3klMjB0byUyMGNyZWF0ZSUyMHRoZSUyMGNoYXJhY3RlcnMuJTBBJTBBVGhlJTIwVHJhbnNmb3JtZXJzJTIwbWVjaGElMjB3ZXJlJTIwbGFyZ2VseSUyMGRlc2lnbmVkJTIwYnklMjBTaCVDNSU4RGppJTIwS2F3YW1vcmklMkMlMjB0aGUlMjBjcmVhdG9yJTIwb2YlMEF0aGUlMjBKYXBhbmVzZSUyMG1lY2hhJTIwYW5pbWUlMjBmcmFuY2hpc2UlMjBNYWNyb3NzJTIwKHdoaWNoJTIwd2FzJTIwYWRhcHRlZCUyMGludG8lMjB0aGUlMjBSb2JvdGVjaCUwQWZyYW5jaGlzZSUyMGluJTIwTm9ydGglMjBBbWVyaWNhKS4lMjBLYXdhbW9yaSUyMGNhbWUlMjB1cCUyMHdpdGglMjB0aGUlMjBpZGVhJTIwb2YlMjB0cmFuc2Zvcm1pbmclMEFtZWNocyUyMHdoaWxlJTIwd29ya2luZyUyMG9uJTIwdGhlJTIwRGlhY2xvbmUlMjBhbmQlMjBNYWNyb3NzJTIwZnJhbmNoaXNlcyUyMGluJTIwdGhlJTIwZWFybHklMjAxOTgwcyUwQShzdWNoJTIwYXMlMjB0aGUlMjBWRi0xJTIwVmFsa3lyaWUlMjBpbiUyME1hY3Jvc3MlMjBhbmQlMjBSb2JvdGVjaCklMkMlMjB3aXRoJTIwaGlzJTIwRGlhY2xvbmUlMjBtZWNocyUwQWxhdGVyJTIwcHJvdmlkaW5nJTIwdGhlJTIwYmFzaXMlMjBmb3IlMjBUcmFuc2Zvcm1lcnMuJTBBJTBBVGhlJTIwcHJpbWFyeSUyMGNvbmNlcHQlMjBvZiUyMEdlbmVyYXRpb24lMjBPbmUlMjBpcyUyMHRoYXQlMjB0aGUlMjBoZXJvaWMlMjBPcHRpbXVzJTIwUHJpbWUlMkMlMjB0aGUlMEF2aWxsYWlub3VzJTIwTWVnYXRyb24lMkMlMjBhbmQlMjB0aGVpciUyMGZpbmVzdCUyMHNvbGRpZXJzJTIwY3Jhc2glMjBsYW5kJTIwb24lMjBwcmUtaGlzdG9yaWMlMjBFYXJ0aCUwQWluJTIwdGhlJTIwQXJrJTIwYW5kJTIwdGhlJTIwTmVtZXNpcyUyMGJlZm9yZSUyMGF3YWtlbmluZyUyMGluJTIwMTk4NSUyQyUyMEN5YmVydHJvbiUyMGh1cnRsaW5nJTIwdGhyb3VnaCUwQXRoZSUyME5ldXRyYWwlMjB6b25lJTIwYXMlMjBhbiUyMGVmZmVjdCUyMG9mJTIwdGhlJTIwd2FyLiUyMFRoZSUyME1hcnZlbCUyMGNvbWljJTIwd2FzJTIwb3JpZ2luYWxseSUyMHBhcnQlMEFvZiUyMHRoZSUyMG1haW4lMjBNYXJ2ZWwlMjBVbml2ZXJzZSUyQyUyMHdpdGglMjBhcHBlYXJhbmNlcyUyMGZyb20lMjBTcGlkZXItTWFuJTIwYW5kJTIwTmljayUyMEZ1cnklMkMlMEFwbHVzJTIwc29tZSUyMGNhbWVvcyUyQyUyMGFzJTIwd2VsbCUyMGFzJTIwYSUyMHZpc2l0JTIwdG8lMjB0aGUlMjBTYXZhZ2UlMjBMYW5kLiUwQSUwQVRoZSUyMFRyYW5zZm9ybWVycyUyMFRWJTIwc2VyaWVzJTIwYmVnYW4lMjBhcm91bmQlMjB0aGUlMjBzYW1lJTIwdGltZS4lMjBQcm9kdWNlZCUyMGJ5JTIwU3VuYm93JTBBUHJvZHVjdGlvbnMlMjBhbmQlMjBNYXJ2ZWwlMjBQcm9kdWN0aW9ucyUyQyUyMGxhdGVyJTIwSGFzYnJvJTIwUHJvZHVjdGlvbnMlMkMlMjBmcm9tJTIwdGhlJTIwc3RhcnQlMjBpdCUwQWNvbnRyYWRpY3RlZCUyMEJ1ZGlhbnNreSdzJTIwYmFja3N0b3JpZXMuJTIwVGhlJTIwVFYlMjBzZXJpZXMlMjBzaG93cyUyMHRoZSUyMEF1dG9ib3RzJTIwbG9va2luZyUwQWZvciUyMG5ldyUyMGVuZXJneSUyMHNvdXJjZXMlMkMlMjBhbmQlMjBjcmFzaCUyMGxhbmRpbmclMjBhcyUyMHRoZSUyMERlY2VwdGljb25zJTIwYXR0YWNrLiUyME1hcnZlbCUwQWludGVycHJldGVkJTIwdGhlJTIwQXV0b2JvdHMlMjBhcyUyMGRlc3Ryb3lpbmclMjBhJTIwcm9ndWUlMjBhc3Rlcm9pZCUyMGFwcHJvYWNoaW5nJTIwQ3liZXJ0cm9uLiUwQVNob2Nrd2F2ZSUyMGlzJTIwbG95YWwlMjB0byUyME1lZ2F0cm9uJTIwaW4lMjB0aGUlMjBUViUyMHNlcmllcyUyQyUyMGtlZXBpbmclMjBDeWJlcnRyb24lMjBpbiUyMGElMEFzdGFsZW1hdGUlMjBkdXJpbmclMjBoaXMlMjBhYnNlbmNlJTJDJTIwYnV0JTIwaW4lMjB0aGUlMjBjb21pYyUyMGJvb2slMjBoZSUyMGF0dGVtcHRzJTIwdG8lMjB0YWtlJTIwY29tbWFuZCUwQW9mJTIwdGhlJTIwRGVjZXB0aWNvbnMuJTIwVGhlJTIwVFYlMjBzZXJpZXMlMjB3b3VsZCUyMGFsc28lMjBkaWZmZXIlMjB3aWxkbHklMjBmcm9tJTIwdGhlJTIwb3JpZ2lucyUwQUJ1ZGlhbnNreSUyMGhhZCUyMGNyZWF0ZWQlMjBmb3IlMjB0aGUlMjBEaW5vYm90cyUyQyUyMHRoZSUyMERlY2VwdGljb24lMjB0dXJuZWQlMjBBdXRvYm90JTIwSmV0ZmlyZSUwQShrbm93biUyMGFzJTIwU2t5ZmlyZSUyMG9uJTIwVFYpJTJDJTIwdGhlJTIwQ29uc3RydWN0aWNvbnMlMjAod2hvJTIwY29tYmluZSUyMHRvJTIwZm9ybSUwQURldmFzdGF0b3IpJTJDJTVCMTklNUQlNUIyMCU1RCUyMGFuZCUyME9tZWdhJTIwU3VwcmVtZS4lMjBUaGUlMjBNYXJ2ZWwlMjBjb21pYyUyMGVzdGFibGlzaGVzJTIwZWFybHklMjBvbiUwQXRoYXQlMjBQcmltZSUyMHdpZWxkcyUyMHRoZSUyMENyZWF0aW9uJTIwTWF0cml4JTJDJTIwd2hpY2glMjBnaXZlcyUyMGxpZmUlMjB0byUyMG1hY2hpbmVzLiUyMEluJTIwdGhlJTBBc2Vjb25kJTIwc2Vhc29uJTJDJTIwdGhlJTIwdHdvLXBhcnQlMjBlcGlzb2RlJTIwVGhlJTIwS2V5JTIwdG8lMjBWZWN0b3IlMjBTaWdtYSUyMGludHJvZHVjZWQlMjB0aGUlMEFhbmNpZW50JTIwVmVjdG9yJTIwU2lnbWElMjBjb21wdXRlciUyQyUyMHdoaWNoJTIwc2VydmVkJTIwdGhlJTIwc2FtZSUyMG9yaWdpbmFsJTIwcHVycG9zZSUyMGFzJTIwdGhlJTBBQ3JlYXRpb24lMjBNYXRyaXglMjAoZ2l2aW5nJTIwbGlmZSUyMHRvJTIwVHJhbnNmb3JtZXJzKSUyQyUyMGFuZCUyMGl0cyUyMGd1YXJkaWFuJTIwQWxwaGElMjBUcmlvbi4lMEElMjIlMjIlMjIlMEElMEFpbnB1dHMlMjAlM0QlMjB0b2tlbml6ZXIodGV4dCUyQyUyMHJldHVybl90ZW5zb3JzJTNEJTIycHQlMjIpJTBBbG9naXRzJTIwJTNEJTIwbW9kZWwoKippbnB1dHMpLmxvZ2l0cw==",highlighted:`text = <span class="hljs-string">&quot;&quot;&quot;
Generation One is a retroactive term for the Transformers characters that
appeared between 1984 and 1993. The Transformers began with the 1980s Japanese
toy lines Micro Change and Diaclone. They presented robots able to transform
into everyday vehicles, electronic items or weapons. Hasbro bought the Micro
Change and Diaclone toys, and partnered with Takara. Marvel Comics was hired by
Hasbro to create the backstory; editor-in-chief Jim Shooter wrote an overall
story, and gave the task of creating the characthers to writer Dennis O&#x27;Neil.
Unhappy with O&#x27;Neil&#x27;s work (although O&#x27;Neil created the name &quot;Optimus Prime&quot;),
Shooter chose Bob Budiansky to create the characters.

The Transformers mecha were largely designed by Shōji Kawamori, the creator of
the Japanese mecha anime franchise Macross (which was adapted into the Robotech
franchise in North America). Kawamori came up with the idea of transforming
mechs while working on the Diaclone and Macross franchises in the early 1980s
(such as the VF-1 Valkyrie in Macross and Robotech), with his Diaclone mechs
later providing the basis for Transformers.

The primary concept of Generation One is that the heroic Optimus Prime, the
villainous Megatron, and their finest soldiers crash land on pre-historic Earth
in the Ark and the Nemesis before awakening in 1985, Cybertron hurtling through
the Neutral zone as an effect of the war. The Marvel comic was originally part
of the main Marvel Universe, with appearances from Spider-Man and Nick Fury,
plus some cameos, as well as a visit to the Savage Land.

The Transformers TV series began around the same time. Produced by Sunbow
Productions and Marvel Productions, later Hasbro Productions, from the start it
contradicted Budiansky&#x27;s backstories. The TV series shows the Autobots looking
for new energy sources, and crash landing as the Decepticons attack. Marvel
interpreted the Autobots as destroying a rogue asteroid approaching Cybertron.
Shockwave is loyal to Megatron in the TV series, keeping Cybertron in a
stalemate during his absence, but in the comic book he attempts to take command
of the Decepticons. The TV series would also differ wildly from the origins
Budiansky had created for the Dinobots, the Decepticon turned Autobot Jetfire
(known as Skyfire on TV), the Constructicons (who combine to form
Devastator),[19][20] and Omega Supreme. The Marvel comic establishes early on
that Prime wields the Creation Matrix, which gives life to machines. In the
second season, the two-part episode The Key to Vector Sigma introduced the
ancient Vector Sigma computer, which served the same original purpose as the
Creation Matrix (giving life to Transformers), and its guardian Alpha Trion.
&quot;&quot;&quot;</span>

inputs = tokenizer(text, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)
logits = model(**inputs).logits`,wrap:!1}}),F=new at({props:{code:"SW5kZXhFcnJvciUzQSUyMGluZGV4JTIwb3V0JTIwb2YlMjByYW5nZSUyMGluJTIwc2VsZg==",highlighted:'IndexError: index out of <span class="hljs-built_in">range</span> <span class="hljs-keyword">in</span> self',wrap:!1}}),L=new re({props:{title:"Choosing a descriptive title",local:"choosing-a-descriptive-title",headingTag:"h3"}}),D=new re({props:{title:"Formatting your code snippets",local:"formatting-your-code-snippets",headingTag:"h3"}}),ee=new re({props:{title:"Including the full traceback",local:"including-the-full-traceback",headingTag:"h3"}}),ae=new re({props:{title:"Providing a reproducible example",local:"providing-a-reproducible-example",headingTag:"h3"}}),{c(){M=n("meta"),me=o(),he=n("p"),ue=o(),h(U.$$.fragment),de=o(),h(j.$$.fragment),pe=o(),h(G.$$.fragment),Me=o(),B=n("p"),B.innerHTML=st,ye=o(),w=n("div"),w.innerHTML=rt,we=o(),I=n("p"),I.innerHTML=ht,fe=o(),k=n("p"),k.innerHTML=ct,be=o(),Z=n("p"),Z.innerHTML=mt,ge=o(),W=n("p"),W.innerHTML=ut,Te=o(),h(H.$$.fragment),Je=o(),x=n("p"),x.textContent=dt,ve=o(),h(V.$$.fragment),Ue=o(),C=n("p"),C.innerHTML=pt,je=o(),h(Y.$$.fragment),Ge=o(),h(F.$$.fragment),Be=o(),S=n("p"),S.innerHTML=Mt,Ie=o(),$=n("p"),$.textContent=yt,ke=o(),f=n("div"),f.innerHTML=wt,Ze=o(),R=n("p"),R.textContent=ft,We=o(),b=n("div"),b.innerHTML=bt,He=o(),X=n("p"),X.textContent=gt,xe=o(),g=n("div"),g.innerHTML=Tt,Ve=o(),N=n("p"),N.textContent=Jt,Ce=o(),z=n("ol"),z.innerHTML=vt,Ye=o(),Q=n("p"),Q.textContent=Ut,Fe=o(),h(L.$$.fragment),Se=o(),E=n("p"),E.innerHTML=jt,$e=o(),_=n("blockquote"),_.innerHTML=Gt,Re=o(),P=n("p"),P.innerHTML=Bt,Xe=o(),q=n("blockquote"),q.innerHTML=It,Ne=o(),A=n("p"),A.textContent=kt,ze=o(),h(D.$$.fragment),Qe=o(),O=n("p"),O.textContent=Zt,Le=o(),T=n("div"),T.innerHTML=Wt,Ee=o(),K=n("p"),K.innerHTML=Ht,_e=o(),h(ee.$$.fragment),Pe=o(),te=n("p"),te.innerHTML=xt,qe=o(),J=n("div"),J.innerHTML=Vt,Ae=o(),le=n("p"),le.textContent=Ct,De=o(),ie=n("blockquote"),ie.innerHTML=Yt,Oe=o(),oe=n("p"),oe.textContent=Ft,Ke=o(),h(ae.$$.fragment),et=o(),ne=n("p"),ne.innerHTML=St,tt=o(),v=n("div"),v.innerHTML=$t,lt=o(),se=n("p"),se.textContent=Rt,it=o(),ce=n("p"),this.h()},l(e){const t=_t("svelte-u9bgzb",document.head);M=s(t,"META",{name:!0,content:!0}),t.forEach(l),me=a(e),he=s(e,"P",{}),Xt(he).forEach(l),ue=a(e),c(U.$$.fragment,e),de=a(e),c(j.$$.fragment,e),pe=a(e),c(G.$$.fragment,e),Me=a(e),B=s(e,"P",{"data-svelte-h":!0}),r(B)!=="svelte-ptl1py"&&(B.innerHTML=st),ye=a(e),w=s(e,"DIV",{class:!0,"data-svelte-h":!0}),r(w)!=="svelte-14n1ttz"&&(w.innerHTML=rt),we=a(e),I=s(e,"P",{"data-svelte-h":!0}),r(I)!=="svelte-1hb3c9d"&&(I.innerHTML=ht),fe=a(e),k=s(e,"P",{"data-svelte-h":!0}),r(k)!=="svelte-1hddgeu"&&(k.innerHTML=ct),be=a(e),Z=s(e,"P",{"data-svelte-h":!0}),r(Z)!=="svelte-1jl0aud"&&(Z.innerHTML=mt),ge=a(e),W=s(e,"P",{"data-svelte-h":!0}),r(W)!=="svelte-kj9fmm"&&(W.innerHTML=ut),Te=a(e),c(H.$$.fragment,e),Je=a(e),x=s(e,"P",{"data-svelte-h":!0}),r(x)!=="svelte-qofqof"&&(x.textContent=dt),ve=a(e),c(V.$$.fragment,e),Ue=a(e),C=s(e,"P",{"data-svelte-h":!0}),r(C)!=="svelte-1pbyuk4"&&(C.innerHTML=pt),je=a(e),c(Y.$$.fragment,e),Ge=a(e),c(F.$$.fragment,e),Be=a(e),S=s(e,"P",{"data-svelte-h":!0}),r(S)!=="svelte-ybbg77"&&(S.innerHTML=Mt),Ie=a(e),$=s(e,"P",{"data-svelte-h":!0}),r($)!=="svelte-uzxshx"&&($.textContent=yt),ke=a(e),f=s(e,"DIV",{class:!0,"data-svelte-h":!0}),r(f)!=="svelte-1guiktj"&&(f.innerHTML=wt),Ze=a(e),R=s(e,"P",{"data-svelte-h":!0}),r(R)!=="svelte-14h0l1f"&&(R.textContent=ft),We=a(e),b=s(e,"DIV",{class:!0,"data-svelte-h":!0}),r(b)!=="svelte-xij8sp"&&(b.innerHTML=bt),He=a(e),X=s(e,"P",{"data-svelte-h":!0}),r(X)!=="svelte-1l00i8q"&&(X.textContent=gt),xe=a(e),g=s(e,"DIV",{class:!0,"data-svelte-h":!0}),r(g)!=="svelte-3f05is"&&(g.innerHTML=Tt),Ve=a(e),N=s(e,"P",{"data-svelte-h":!0}),r(N)!=="svelte-ijbi50"&&(N.textContent=Jt),Ce=a(e),z=s(e,"OL",{"data-svelte-h":!0}),r(z)!=="svelte-1muguxp"&&(z.innerHTML=vt),Ye=a(e),Q=s(e,"P",{"data-svelte-h":!0}),r(Q)!=="svelte-1z0sqgv"&&(Q.textContent=Ut),Fe=a(e),c(L.$$.fragment,e),Se=a(e),E=s(e,"P",{"data-svelte-h":!0}),r(E)!=="svelte-182n17j"&&(E.innerHTML=jt),$e=a(e),_=s(e,"BLOCKQUOTE",{"data-svelte-h":!0}),r(_)!=="svelte-qp1hky"&&(_.innerHTML=Gt),Re=a(e),P=s(e,"P",{"data-svelte-h":!0}),r(P)!=="svelte-17fxjte"&&(P.innerHTML=Bt),Xe=a(e),q=s(e,"BLOCKQUOTE",{"data-svelte-h":!0}),r(q)!=="svelte-15dhjoh"&&(q.innerHTML=It),Ne=a(e),A=s(e,"P",{"data-svelte-h":!0}),r(A)!=="svelte-oc83gq"&&(A.textContent=kt),ze=a(e),c(D.$$.fragment,e),Qe=a(e),O=s(e,"P",{"data-svelte-h":!0}),r(O)!=="svelte-1a2ujz5"&&(O.textContent=Zt),Le=a(e),T=s(e,"DIV",{class:!0,"data-svelte-h":!0}),r(T)!=="svelte-pol4zs"&&(T.innerHTML=Wt),Ee=a(e),K=s(e,"P",{"data-svelte-h":!0}),r(K)!=="svelte-36ehy7"&&(K.innerHTML=Ht),_e=a(e),c(ee.$$.fragment,e),Pe=a(e),te=s(e,"P",{"data-svelte-h":!0}),r(te)!=="svelte-1weouup"&&(te.innerHTML=xt),qe=a(e),J=s(e,"DIV",{class:!0,"data-svelte-h":!0}),r(J)!=="svelte-1oxns09"&&(J.innerHTML=Vt),Ae=a(e),le=s(e,"P",{"data-svelte-h":!0}),r(le)!=="svelte-1qce4bu"&&(le.textContent=Ct),De=a(e),ie=s(e,"BLOCKQUOTE",{"data-svelte-h":!0}),r(ie)!=="svelte-1cf7hkp"&&(ie.innerHTML=Yt),Oe=a(e),oe=s(e,"P",{"data-svelte-h":!0}),r(oe)!=="svelte-1c52bzg"&&(oe.textContent=Ft),Ke=a(e),c(ae.$$.fragment,e),et=a(e),ne=s(e,"P",{"data-svelte-h":!0}),r(ne)!=="svelte-7jc688"&&(ne.innerHTML=St),tt=a(e),v=s(e,"DIV",{class:!0,"data-svelte-h":!0}),r(v)!=="svelte-1ckd2c8"&&(v.innerHTML=$t),lt=a(e),se=s(e,"P",{"data-svelte-h":!0}),r(se)!=="svelte-10iqq25"&&(se.textContent=Rt),it=a(e),ce=s(e,"P",{}),Xt(ce).forEach(l),this.h()},h(){y(M,"name","hf:doc:metadata"),y(M,"content",Ot),y(w,"class","flex justify-center"),y(f,"class","flex justify-center"),y(b,"class","flex justify-center"),y(g,"class","flex justify-center"),y(T,"class","flex justify-center"),y(J,"class","flex justify-center"),y(v,"class","flex justify-center")},m(e,t){Pt(document.head,M),i(e,me,t),i(e,he,t),i(e,ue,t),m(U,e,t),i(e,de,t),m(j,e,t),i(e,pe,t),m(G,e,t),i(e,Me,t),i(e,B,t),i(e,ye,t),i(e,w,t),i(e,we,t),i(e,I,t),i(e,fe,t),i(e,k,t),i(e,be,t),i(e,Z,t),i(e,ge,t),i(e,W,t),i(e,Te,t),m(H,e,t),i(e,Je,t),i(e,x,t),i(e,ve,t),m(V,e,t),i(e,Ue,t),i(e,C,t),i(e,je,t),m(Y,e,t),i(e,Ge,t),m(F,e,t),i(e,Be,t),i(e,S,t),i(e,Ie,t),i(e,$,t),i(e,ke,t),i(e,f,t),i(e,Ze,t),i(e,R,t),i(e,We,t),i(e,b,t),i(e,He,t),i(e,X,t),i(e,xe,t),i(e,g,t),i(e,Ve,t),i(e,N,t),i(e,Ce,t),i(e,z,t),i(e,Ye,t),i(e,Q,t),i(e,Fe,t),m(L,e,t),i(e,Se,t),i(e,E,t),i(e,$e,t),i(e,_,t),i(e,Re,t),i(e,P,t),i(e,Xe,t),i(e,q,t),i(e,Ne,t),i(e,A,t),i(e,ze,t),m(D,e,t),i(e,Qe,t),i(e,O,t),i(e,Le,t),i(e,T,t),i(e,Ee,t),i(e,K,t),i(e,_e,t),m(ee,e,t),i(e,Pe,t),i(e,te,t),i(e,qe,t),i(e,J,t),i(e,Ae,t),i(e,le,t),i(e,De,t),i(e,ie,t),i(e,Oe,t),i(e,oe,t),i(e,Ke,t),m(ae,e,t),i(e,et,t),i(e,ne,t),i(e,tt,t),i(e,v,t),i(e,lt,t),i(e,se,t),i(e,it,t),i(e,ce,t),ot=!0},p:zt,i(e){ot||(u(U.$$.fragment,e),u(j.$$.fragment,e),u(G.$$.fragment,e),u(H.$$.fragment,e),u(V.$$.fragment,e),u(Y.$$.fragment,e),u(F.$$.fragment,e),u(L.$$.fragment,e),u(D.$$.fragment,e),u(ee.$$.fragment,e),u(ae.$$.fragment,e),ot=!0)},o(e){d(U.$$.fragment,e),d(j.$$.fragment,e),d(G.$$.fragment,e),d(H.$$.fragment,e),d(V.$$.fragment,e),d(Y.$$.fragment,e),d(F.$$.fragment,e),d(L.$$.fragment,e),d(D.$$.fragment,e),d(ee.$$.fragment,e),d(ae.$$.fragment,e),ot=!1},d(e){e&&(l(me),l(he),l(ue),l(de),l(pe),l(Me),l(B),l(ye),l(w),l(we),l(I),l(fe),l(k),l(be),l(Z),l(ge),l(W),l(Te),l(Je),l(x),l(ve),l(Ue),l(C),l(je),l(Ge),l(Be),l(S),l(Ie),l($),l(ke),l(f),l(Ze),l(R),l(We),l(b),l(He),l(X),l(xe),l(g),l(Ve),l(N),l(Ce),l(z),l(Ye),l(Q),l(Fe),l(Se),l(E),l($e),l(_),l(Re),l(P),l(Xe),l(q),l(Ne),l(A),l(ze),l(Qe),l(O),l(Le),l(T),l(Ee),l(K),l(_e),l(Pe),l(te),l(qe),l(J),l(Ae),l(le),l(De),l(ie),l(Oe),l(oe),l(Ke),l(et),l(ne),l(tt),l(v),l(lt),l(se),l(it),l(ce)),l(M),p(U,e),p(j,e),p(G,e),p(H,e),p(V,e),p(Y,e),p(F,e),p(L,e),p(D,e),p(ee,e),p(ae,e)}}}const Ot='{"title":"Asking for help on the forums","local":"asking-for-help-on-the-forums","sections":[{"title":"Writing a good forum post","local":"writing-a-good-forum-post","sections":[{"title":"Choosing a descriptive title","local":"choosing-a-descriptive-title","sections":[],"depth":3},{"title":"Formatting your code snippets","local":"formatting-your-code-snippets","sections":[],"depth":3},{"title":"Including the full traceback","local":"including-the-full-traceback","sections":[],"depth":3},{"title":"Providing a reproducible example","local":"providing-a-reproducible-example","sections":[],"depth":3}],"depth":2}],"depth":1}';function Kt(nt){return Qt(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class nl extends Lt{constructor(M){super(),Et(this,M,Kt,Dt,Nt,{})}}export{nl as component};
