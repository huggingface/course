import{s as $t,o as Ut,n as vt}from"../chunks/scheduler.37c15a92.js";import{S as jt,i as Rt,g as o,s as n,r as d,A as Lt,h as r,f as l,c as i,j as bt,u as f,x as m,k as Jt,y as Zt,a,v as u,d as c,t as h,w as g,m as Ct,n as kt}from"../chunks/index.2bf4358c.js";import{T as Ne}from"../chunks/Tip.363c041f.js";import{C as re}from"../chunks/CodeBlock.4f5fc1ad.js";import{C as _t}from"../chunks/CourseFloatingBanner.15ba07e6.js";import{H as y}from"../chunks/Heading.8ada512a.js";function At(w){let s;return{c(){s=Ct("When implementing PEFT methods, start with small rank values (4-8) for LoRA and monitor training loss. Use validation sets to prevent overfitting and compare results with full fine-tuning baselines when possible. The effectiveness of different methods can vary by task, so experimentation is key.")},l(M){s=kt(M,"When implementing PEFT methods, start with small rank values (4-8) for LoRA and monitor training loss. Use validation sets to prevent overfitting and compare results with full fine-tuning baselines when possible. The effectiveness of different methods can vary by task, so experimentation is key.")},m(M,p){a(M,s,p)},d(M){M&&l(s)}}}function Gt(w){let s,M="✏️ <strong>Try it out!</strong> Build on your fine-tuned model from the previous section, but fine-tune it with LoRA. Use the <code>HuggingFaceTB/smoltalk</code> dataset to fine-tune a <code>deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B</code> model, using the LoRA configuration we defined above.";return{c(){s=o("p"),s.innerHTML=M},l(p){s=r(p,"P",{"data-svelte-h":!0}),m(s)!=="svelte-loec6q"&&(s.innerHTML=M)},m(p,T){a(p,s,T)},p:vt,d(p){p&&l(s)}}}function Bt(w){let s,M="✏️ <strong>Try it out!</strong> Merge the adapter weights back into the base model. Use the <code>HuggingFaceTB/smoltalk</code> dataset to fine-tune a <code>deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B</code> model, using the LoRA configuration we defined above.";return{c(){s=o("p"),s.innerHTML=M},l(p){s=r(p,"P",{"data-svelte-h":!0}),m(s)!=="svelte-tjvbvt"&&(s.innerHTML=M)},m(p,T){a(p,s,T)},p:vt,d(p){p&&l(s)}}}function It(w){let s,M,p,T,$,me,U,pe,j,De="Fine-tuning large language models is a resource intensive process. LoRA is a technique that allows us to fine-tune large language models with a small number of parameters. It works by adding and optimizing smaller matrices to the attention weights, typically reducing trainable parameters by about 90%.",de,R,fe,L,Ke='LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that freezes the pre-trained model weights and injects trainable rank decomposition matrices into the model’s layers. Instead of training all model parameters during fine-tuning, LoRA decomposes the weight updates into smaller matrices through low-rank decomposition, significantly reducing the number of trainable parameters while maintaining model performance. For example, when applied to GPT-3 175B, LoRA reduced trainable parameters by 10,000x and GPU memory requirements by 3x compared to full fine-tuning. You can read more about LoRA in the <a href="https://arxiv.org/pdf/2106.09685" rel="nofollow">LoRA paper</a>.',ue,Z,Oe="LoRA works by adding pairs of rank decomposition matrices to transformer layers, typically focusing on attention weights. During inference, these adapter weights can be merged with the base model, resulting in no additional latency overhead. LoRA is particularly useful for adapting large language models to specific tasks or domains while keeping resource requirements manageable.",ce,C,he,k,et="<li><p><strong>Memory Efficiency</strong>:</p> <ul><li>Only adapter parameters are stored in GPU memory</li> <li>Base model weights remain frozen and can be loaded in lower precision</li> <li>Enables fine-tuning of large models on consumer GPUs</li></ul></li> <li><p><strong>Training Features</strong>:</p> <ul><li>Native PEFT/LoRA integration with minimal setup</li> <li>Support for QLoRA (Quantized LoRA) for even better memory efficiency</li></ul></li> <li><p><strong>Adapter Management</strong>:</p> <ul><li>Adapter weight saving during checkpoints</li> <li>Features to merge adapters back into base model</li></ul></li>",ge,_,Me,A,tt='<a href="https://github.com/huggingface/peft" rel="nofollow">PEFT</a> is a library that provides a unified interface for loading and managing PEFT methods, including LoRA. It allows you to easily load and switch between different PEFT methods, making it easier to experiment with different fine-tuning techniques.',ye,G,lt="Adapters can be loaded onto a pretrained model with <code>load_adapter()</code>, which is useful for trying out different adapters whose weights aren’t merged. Set the active adapter weights with the <code>set_adapter()</code> function. To return the base model, you could use unload() to unload all of the LoRA modules. This makes it easy to switch between different task-specific weights.",we,B,Te,I,at='<img src="https://github.com/huggingface/smol-course/raw/main/3_parameter_efficient_finetuning/images/lora_adapter.png" alt="lora_load_adapter"/>',be,E,Je,F,nt='The <a href="https://huggingface.co/docs/trl/sft_trainer" rel="nofollow">SFTTrainer</a> from <code>trl</code> provides integration with LoRA adapters through the <a href="https://huggingface.co/docs/peft/en/index" rel="nofollow">PEFT</a> library. This means that we can fine-tune a model in the same way as we did with SFT, but use LoRA to reduce the number of parameters we need to train.',ve,x,it="We’ll use the <code>LoRAConfig</code> class from PEFT in our example. The setup requires just a few configuration steps:",$e,W,st="<li>Define the LoRA configuration (rank, alpha, dropout)</li> <li>Create the SFTTrainer with PEFT config</li> <li>Train and save the adapter weights</li>",Ue,H,je,V,ot="Let’s walk through the LoRA configuration and key parameters.",Re,X,rt="<thead><tr><th>Parameter</th> <th>Description</th></tr></thead> <tbody><tr><td><code>r</code> (rank)</td> <td>Dimension of the low-rank matrices used for weight updates. Typically between 4-32. Lower values provide more compression but potentially less expressiveness.</td></tr> <tr><td><code>lora_alpha</code></td> <td>Scaling factor for LoRA layers, usually set to 2x the rank value. Higher values result in stronger adaptation effects.</td></tr> <tr><td><code>lora_dropout</code></td> <td>Dropout probability for LoRA layers, typically 0.05-0.1. Higher values help prevent overfitting during training.</td></tr> <tr><td><code>bias</code></td> <td>Controls training of bias terms. Options are “none”, “all”, or “lora_only”. “none” is most common for memory efficiency.</td></tr> <tr><td><code>target_modules</code></td> <td>Specifies which model modules to apply LoRA to. Can be “all-linear” or specific modules like “q_proj,v_proj”. More modules enable greater adaptability but increase memory usage.</td></tr></tbody>",Le,b,Ze,Q,Ce,Y,mt="PEFT methods can be combined with TRL for fine-tuning to reduce memory requirements. We can pass the  <code>LoraConfig</code> to the model when loading it.",ke,S,_e,P,pt="Above, we used <code>device_map=&quot;auto&quot;</code> to automatically assign the model to the correct device. You can also manually assign the model to a specific device using <code>device_map={&quot;&quot;: device_index}</code>.",Ae,z,dt="We will also need to define the <code>SFTTrainer</code> with the LoRA configuration.",Ge,q,Be,J,Ie,N,Ee,D,ft="After training with LoRA, you might want to merge the adapter weights back into the base model for easier deployment. This creates a single model with the combined weights, eliminating the need to load adapters separately during inference.",Fe,K,ut="The merging process requires attention to memory management and precision. Since you’ll need to load both the base model and adapter weights simultaneously, ensure sufficient GPU/CPU memory is available. Using <code>device_map=&quot;auto&quot;</code> in <code>transformers</code> will find the correct device for the model based on your hardware.",xe,O,ct="Maintain consistent precision (e.g., float16) throughout the process, matching the precision used during training and saving the merged model in the same format for deployment.",We,ee,He,te,ht="After training a LoRA adapter, you can merge the adapter weights back into the base model. Here’s how to do it:",Ve,le,Xe,ae,gt="If you encounter size discrepancies in the saved model, ensure you’re also saving the tokenizer:",Qe,ne,Ye,v,Se,ie,Pe,se,Mt='<li><a href="https://arxiv.org/pdf/2106.09685" rel="nofollow">LoRA: Low-Rank Adaptation of Large Language Models</a></li> <li><a href="https://huggingface.co/docs/peft" rel="nofollow">PEFT Documentation</a></li> <li><a href="https://huggingface.co/blog/peft" rel="nofollow">Hugging Face blog post on PEFT</a></li>',ze,oe,qe;return $=new _t({props:{chapter:2,classNames:"absolute z-10 right-0 top-0",notebooks:[{label:"Google Colab",value:"https://colab.research.google.com/github/huggingface/notebooks/blob/main/course/en/chapter11/section4.ipynb"}]}}),U=new y({props:{title:"LoRA (Low-Rank Adaptation)",local:"lora-low-rank-adaptation",headingTag:"h1"}}),R=new y({props:{title:"Understanding LoRA",local:"understanding-lora",headingTag:"h2"}}),C=new y({props:{title:"Key advantages of LoRA",local:"key-advantages-of-lora",headingTag:"h2"}}),_=new y({props:{title:"Loading LoRA Adapters with PEFT",local:"loading-lora-adapters-with-peft",headingTag:"h2"}}),B=new re({props:{code:"ZnJvbSUyMHBlZnQlMjBpbXBvcnQlMjBQZWZ0TW9kZWwlMkMlMjBQZWZ0Q29uZmlnJTBBJTBBY29uZmlnJTIwJTNEJTIwUGVmdENvbmZpZy5mcm9tX3ByZXRyYWluZWQoJTIyeWJlbGthZGElMkZvcHQtMzUwbS1sb3JhJTIyKSUwQW1vZGVsJTIwJTNEJTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0uZnJvbV9wcmV0cmFpbmVkKGNvbmZpZy5iYXNlX21vZGVsX25hbWVfb3JfcGF0aCklMEFsb3JhX21vZGVsJTIwJTNEJTIwUGVmdE1vZGVsLmZyb21fcHJldHJhaW5lZChtb2RlbCUyQyUyMCUyMnliZWxrYWRhJTJGb3B0LTM1MG0tbG9yYSUyMik=",highlighted:`<span class="hljs-keyword">from</span> peft <span class="hljs-keyword">import</span> PeftModel, PeftConfig

config = PeftConfig.from_pretrained(<span class="hljs-string">&quot;ybelkada/opt-350m-lora&quot;</span>)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
lora_model = PeftModel.from_pretrained(model, <span class="hljs-string">&quot;ybelkada/opt-350m-lora&quot;</span>)`,wrap:!1}}),E=new y({props:{title:"Fine-tune LLM using trl and the SFTTrainer with LoRA",local:"fine-tune-llm-using-trl-and-the-sfttrainer-with-lora",headingTag:"h2"}}),H=new y({props:{title:"LoRA Configuration",local:"lora-configuration",headingTag:"h2"}}),b=new Ne({props:{$$slots:{default:[At]},$$scope:{ctx:w}}}),Q=new y({props:{title:"Using TRL with PEFT",local:"using-trl-with-peft",headingTag:"h2"}}),S=new re({props:{code:"ZnJvbSUyMHBlZnQlMjBpbXBvcnQlMjBMb3JhQ29uZmlnJTBBJTBBJTIzJTIwVE9ETyUzQSUyMENvbmZpZ3VyZSUyMExvUkElMjBwYXJhbWV0ZXJzJTBBJTIzJTIwciUzQSUyMHJhbmslMjBkaW1lbnNpb24lMjBmb3IlMjBMb1JBJTIwdXBkYXRlJTIwbWF0cmljZXMlMjAoc21hbGxlciUyMCUzRCUyMG1vcmUlMjBjb21wcmVzc2lvbiklMEFyYW5rX2RpbWVuc2lvbiUyMCUzRCUyMDYlMEElMjMlMjBsb3JhX2FscGhhJTNBJTIwc2NhbGluZyUyMGZhY3RvciUyMGZvciUyMExvUkElMjBsYXllcnMlMjAoaGlnaGVyJTIwJTNEJTIwc3Ryb25nZXIlMjBhZGFwdGF0aW9uKSUwQWxvcmFfYWxwaGElMjAlM0QlMjA4JTBBJTIzJTIwbG9yYV9kcm9wb3V0JTNBJTIwZHJvcG91dCUyMHByb2JhYmlsaXR5JTIwZm9yJTIwTG9SQSUyMGxheWVycyUyMChoZWxwcyUyMHByZXZlbnQlMjBvdmVyZml0dGluZyklMEFsb3JhX2Ryb3BvdXQlMjAlM0QlMjAwLjA1JTBBJTBBcGVmdF9jb25maWclMjAlM0QlMjBMb3JhQ29uZmlnKCUwQSUyMCUyMCUyMCUyMHIlM0RyYW5rX2RpbWVuc2lvbiUyQyUyMCUyMCUyMyUyMFJhbmslMjBkaW1lbnNpb24lMjAtJTIwdHlwaWNhbGx5JTIwYmV0d2VlbiUyMDQtMzIlMEElMjAlMjAlMjAlMjBsb3JhX2FscGhhJTNEbG9yYV9hbHBoYSUyQyUyMCUyMCUyMyUyMExvUkElMjBzY2FsaW5nJTIwZmFjdG9yJTIwLSUyMHR5cGljYWxseSUyMDJ4JTIwcmFuayUwQSUyMCUyMCUyMCUyMGxvcmFfZHJvcG91dCUzRGxvcmFfZHJvcG91dCUyQyUyMCUyMCUyMyUyMERyb3BvdXQlMjBwcm9iYWJpbGl0eSUyMGZvciUyMExvUkElMjBsYXllcnMlMEElMjAlMjAlMjAlMjBiaWFzJTNEJTIybm9uZSUyMiUyQyUyMCUyMCUyMyUyMEJpYXMlMjB0eXBlJTIwZm9yJTIwTG9SQS4lMjB0aGUlMjBjb3JyZXNwb25kaW5nJTIwYmlhc2VzJTIwd2lsbCUyMGJlJTIwdXBkYXRlZCUyMGR1cmluZyUyMHRyYWluaW5nLiUwQSUyMCUyMCUyMCUyMHRhcmdldF9tb2R1bGVzJTNEJTIyYWxsLWxpbmVhciUyMiUyQyUyMCUyMCUyMyUyMFdoaWNoJTIwbW9kdWxlcyUyMHRvJTIwYXBwbHklMjBMb1JBJTIwdG8lMEElMjAlMjAlMjAlMjB0YXNrX3R5cGUlM0QlMjJDQVVTQUxfTE0lMjIlMkMlMjAlMjAlMjMlMjBUYXNrJTIwdHlwZSUyMGZvciUyMG1vZGVsJTIwYXJjaGl0ZWN0dXJlJTBBKQ==",highlighted:`<span class="hljs-keyword">from</span> peft <span class="hljs-keyword">import</span> LoraConfig

<span class="hljs-comment"># <span class="hljs-doctag">TODO:</span> Configure LoRA parameters</span>
<span class="hljs-comment"># r: rank dimension for LoRA update matrices (smaller = more compression)</span>
rank_dimension = <span class="hljs-number">6</span>
<span class="hljs-comment"># lora_alpha: scaling factor for LoRA layers (higher = stronger adaptation)</span>
lora_alpha = <span class="hljs-number">8</span>
<span class="hljs-comment"># lora_dropout: dropout probability for LoRA layers (helps prevent overfitting)</span>
lora_dropout = <span class="hljs-number">0.05</span>

peft_config = LoraConfig(
    r=rank_dimension,  <span class="hljs-comment"># Rank dimension - typically between 4-32</span>
    lora_alpha=lora_alpha,  <span class="hljs-comment"># LoRA scaling factor - typically 2x rank</span>
    lora_dropout=lora_dropout,  <span class="hljs-comment"># Dropout probability for LoRA layers</span>
    bias=<span class="hljs-string">&quot;none&quot;</span>,  <span class="hljs-comment"># Bias type for LoRA. the corresponding biases will be updated during training.</span>
    target_modules=<span class="hljs-string">&quot;all-linear&quot;</span>,  <span class="hljs-comment"># Which modules to apply LoRA to</span>
    task_type=<span class="hljs-string">&quot;CAUSAL_LM&quot;</span>,  <span class="hljs-comment"># Task type for model architecture</span>
)`,wrap:!1}}),q=new re({props:{code:"JTIzJTIwQ3JlYXRlJTIwU0ZUVHJhaW5lciUyMHdpdGglMjBMb1JBJTIwY29uZmlndXJhdGlvbiUwQXRyYWluZXIlMjAlM0QlMjBTRlRUcmFpbmVyKCUwQSUyMCUyMCUyMCUyMG1vZGVsJTNEbW9kZWwlMkMlMEElMjAlMjAlMjAlMjBhcmdzJTNEYXJncyUyQyUwQSUyMCUyMCUyMCUyMHRyYWluX2RhdGFzZXQlM0RkYXRhc2V0JTVCJTIydHJhaW4lMjIlNUQlMkMlMEElMjAlMjAlMjAlMjBwZWZ0X2NvbmZpZyUzRHBlZnRfY29uZmlnJTJDJTIwJTIwJTIzJTIwTG9SQSUyMGNvbmZpZ3VyYXRpb24lMEElMjAlMjAlMjAlMjBtYXhfc2VxX2xlbmd0aCUzRG1heF9zZXFfbGVuZ3RoJTJDJTIwJTIwJTIzJTIwTWF4aW11bSUyMHNlcXVlbmNlJTIwbGVuZ3RoJTBBJTIwJTIwJTIwJTIwcHJvY2Vzc2luZ19jbGFzcyUzRHRva2VuaXplciUyQyUwQSk=",highlighted:`<span class="hljs-comment"># Create SFTTrainer with LoRA configuration</span>
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset[<span class="hljs-string">&quot;train&quot;</span>],
    peft_config=peft_config,  <span class="hljs-comment"># LoRA configuration</span>
    max_seq_length=max_seq_length,  <span class="hljs-comment"># Maximum sequence length</span>
    processing_class=tokenizer,
)`,wrap:!1}}),J=new Ne({props:{$$slots:{default:[Gt]},$$scope:{ctx:w}}}),N=new y({props:{title:"Merging LoRA Adapters",local:"merging-lora-adapters",headingTag:"h2"}}),ee=new y({props:{title:"Merging Implementation",local:"merging-implementation",headingTag:"h2"}}),le=new re({props:{code:"aW1wb3J0JTIwdG9yY2glMEFmcm9tJTIwdHJhbnNmb3JtZXJzJTIwaW1wb3J0JTIwQXV0b01vZGVsRm9yQ2F1c2FsTE0lMEFmcm9tJTIwcGVmdCUyMGltcG9ydCUyMFBlZnRNb2RlbCUwQSUwQSUyMyUyMDEuJTIwTG9hZCUyMHRoZSUyMGJhc2UlMjBtb2RlbCUwQWJhc2VfbW9kZWwlMjAlM0QlMjBBdXRvTW9kZWxGb3JDYXVzYWxMTS5mcm9tX3ByZXRyYWluZWQoJTBBJTIwJTIwJTIwJTIwJTIyYmFzZV9tb2RlbF9uYW1lJTIyJTJDJTIwdG9yY2hfZHR5cGUlM0R0b3JjaC5mbG9hdDE2JTJDJTIwZGV2aWNlX21hcCUzRCUyMmF1dG8lMjIlMEEpJTBBJTBBJTIzJTIwMi4lMjBMb2FkJTIwdGhlJTIwUEVGVCUyMG1vZGVsJTIwd2l0aCUyMGFkYXB0ZXIlMEFwZWZ0X21vZGVsJTIwJTNEJTIwUGVmdE1vZGVsLmZyb21fcHJldHJhaW5lZCglMEElMjAlMjAlMjAlMjBiYXNlX21vZGVsJTJDJTIwJTIycGF0aCUyRnRvJTJGYWRhcHRlciUyMiUyQyUyMHRvcmNoX2R0eXBlJTNEdG9yY2guZmxvYXQxNiUwQSklMEElMEElMjMlMjAzLiUyME1lcmdlJTIwYWRhcHRlciUyMHdlaWdodHMlMjB3aXRoJTIwYmFzZSUyMG1vZGVsJTBBbWVyZ2VkX21vZGVsJTIwJTNEJTIwcGVmdF9tb2RlbC5tZXJnZV9hbmRfdW5sb2FkKCk=",highlighted:`<span class="hljs-keyword">import</span> torch
<span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoModelForCausalLM
<span class="hljs-keyword">from</span> peft <span class="hljs-keyword">import</span> PeftModel

<span class="hljs-comment"># 1. Load the base model</span>
base_model = AutoModelForCausalLM.from_pretrained(
    <span class="hljs-string">&quot;base_model_name&quot;</span>, torch_dtype=torch.float16, device_map=<span class="hljs-string">&quot;auto&quot;</span>
)

<span class="hljs-comment"># 2. Load the PEFT model with adapter</span>
peft_model = PeftModel.from_pretrained(
    base_model, <span class="hljs-string">&quot;path/to/adapter&quot;</span>, torch_dtype=torch.float16
)

<span class="hljs-comment"># 3. Merge adapter weights with base model</span>
merged_model = peft_model.merge_and_unload()`,wrap:!1}}),ne=new re({props:{code:"JTIzJTIwU2F2ZSUyMGJvdGglMjBtb2RlbCUyMGFuZCUyMHRva2VuaXplciUwQXRva2VuaXplciUyMCUzRCUyMEF1dG9Ub2tlbml6ZXIuZnJvbV9wcmV0cmFpbmVkKCUyMmJhc2VfbW9kZWxfbmFtZSUyMiklMEFtZXJnZWRfbW9kZWwuc2F2ZV9wcmV0cmFpbmVkKCUyMnBhdGglMkZ0byUyRnNhdmUlMkZtZXJnZWRfbW9kZWwlMjIpJTBBdG9rZW5pemVyLnNhdmVfcHJldHJhaW5lZCglMjJwYXRoJTJGdG8lMkZzYXZlJTJGbWVyZ2VkX21vZGVsJTIyKQ==",highlighted:`<span class="hljs-comment"># Save both model and tokenizer</span>
tokenizer = AutoTokenizer.from_pretrained(<span class="hljs-string">&quot;base_model_name&quot;</span>)
merged_model.save_pretrained(<span class="hljs-string">&quot;path/to/save/merged_model&quot;</span>)
tokenizer.save_pretrained(<span class="hljs-string">&quot;path/to/save/merged_model&quot;</span>)`,wrap:!1}}),v=new Ne({props:{$$slots:{default:[Bt]},$$scope:{ctx:w}}}),ie=new y({props:{title:"Resources",local:"resources",headingTag:"h1"}}),{c(){s=o("meta"),M=n(),p=o("p"),T=n(),d($.$$.fragment),me=n(),d(U.$$.fragment),pe=n(),j=o("p"),j.textContent=De,de=n(),d(R.$$.fragment),fe=n(),L=o("p"),L.innerHTML=Ke,ue=n(),Z=o("p"),Z.textContent=Oe,ce=n(),d(C.$$.fragment),he=n(),k=o("ol"),k.innerHTML=et,ge=n(),d(_.$$.fragment),Me=n(),A=o("p"),A.innerHTML=tt,ye=n(),G=o("p"),G.innerHTML=lt,we=n(),d(B.$$.fragment),Te=n(),I=o("p"),I.innerHTML=at,be=n(),d(E.$$.fragment),Je=n(),F=o("p"),F.innerHTML=nt,ve=n(),x=o("p"),x.innerHTML=it,$e=n(),W=o("ol"),W.innerHTML=st,Ue=n(),d(H.$$.fragment),je=n(),V=o("p"),V.textContent=ot,Re=n(),X=o("table"),X.innerHTML=rt,Le=n(),d(b.$$.fragment),Ze=n(),d(Q.$$.fragment),Ce=n(),Y=o("p"),Y.innerHTML=mt,ke=n(),d(S.$$.fragment),_e=n(),P=o("p"),P.innerHTML=pt,Ae=n(),z=o("p"),z.innerHTML=dt,Ge=n(),d(q.$$.fragment),Be=n(),d(J.$$.fragment),Ie=n(),d(N.$$.fragment),Ee=n(),D=o("p"),D.textContent=ft,Fe=n(),K=o("p"),K.innerHTML=ut,xe=n(),O=o("p"),O.textContent=ct,We=n(),d(ee.$$.fragment),He=n(),te=o("p"),te.textContent=ht,Ve=n(),d(le.$$.fragment),Xe=n(),ae=o("p"),ae.textContent=gt,Qe=n(),d(ne.$$.fragment),Ye=n(),d(v.$$.fragment),Se=n(),d(ie.$$.fragment),Pe=n(),se=o("ul"),se.innerHTML=Mt,ze=n(),oe=o("p"),this.h()},l(e){const t=Lt("svelte-u9bgzb",document.head);s=r(t,"META",{name:!0,content:!0}),t.forEach(l),M=i(e),p=r(e,"P",{}),bt(p).forEach(l),T=i(e),f($.$$.fragment,e),me=i(e),f(U.$$.fragment,e),pe=i(e),j=r(e,"P",{"data-svelte-h":!0}),m(j)!=="svelte-25es4k"&&(j.textContent=De),de=i(e),f(R.$$.fragment,e),fe=i(e),L=r(e,"P",{"data-svelte-h":!0}),m(L)!=="svelte-1nturj5"&&(L.innerHTML=Ke),ue=i(e),Z=r(e,"P",{"data-svelte-h":!0}),m(Z)!=="svelte-1ore9lv"&&(Z.textContent=Oe),ce=i(e),f(C.$$.fragment,e),he=i(e),k=r(e,"OL",{"data-svelte-h":!0}),m(k)!=="svelte-lq2nv5"&&(k.innerHTML=et),ge=i(e),f(_.$$.fragment,e),Me=i(e),A=r(e,"P",{"data-svelte-h":!0}),m(A)!=="svelte-1740kfn"&&(A.innerHTML=tt),ye=i(e),G=r(e,"P",{"data-svelte-h":!0}),m(G)!=="svelte-5pfbou"&&(G.innerHTML=lt),we=i(e),f(B.$$.fragment,e),Te=i(e),I=r(e,"P",{"data-svelte-h":!0}),m(I)!=="svelte-bkhm6l"&&(I.innerHTML=at),be=i(e),f(E.$$.fragment,e),Je=i(e),F=r(e,"P",{"data-svelte-h":!0}),m(F)!=="svelte-1j7r8p4"&&(F.innerHTML=nt),ve=i(e),x=r(e,"P",{"data-svelte-h":!0}),m(x)!=="svelte-1k0uws"&&(x.innerHTML=it),$e=i(e),W=r(e,"OL",{"data-svelte-h":!0}),m(W)!=="svelte-872dww"&&(W.innerHTML=st),Ue=i(e),f(H.$$.fragment,e),je=i(e),V=r(e,"P",{"data-svelte-h":!0}),m(V)!=="svelte-mvlx3u"&&(V.textContent=ot),Re=i(e),X=r(e,"TABLE",{"data-svelte-h":!0}),m(X)!=="svelte-cgbomc"&&(X.innerHTML=rt),Le=i(e),f(b.$$.fragment,e),Ze=i(e),f(Q.$$.fragment,e),Ce=i(e),Y=r(e,"P",{"data-svelte-h":!0}),m(Y)!=="svelte-1ddnald"&&(Y.innerHTML=mt),ke=i(e),f(S.$$.fragment,e),_e=i(e),P=r(e,"P",{"data-svelte-h":!0}),m(P)!=="svelte-d19z9v"&&(P.innerHTML=pt),Ae=i(e),z=r(e,"P",{"data-svelte-h":!0}),m(z)!=="svelte-1rh9axm"&&(z.innerHTML=dt),Ge=i(e),f(q.$$.fragment,e),Be=i(e),f(J.$$.fragment,e),Ie=i(e),f(N.$$.fragment,e),Ee=i(e),D=r(e,"P",{"data-svelte-h":!0}),m(D)!=="svelte-159k8bc"&&(D.textContent=ft),Fe=i(e),K=r(e,"P",{"data-svelte-h":!0}),m(K)!=="svelte-1lctb1u"&&(K.innerHTML=ut),xe=i(e),O=r(e,"P",{"data-svelte-h":!0}),m(O)!=="svelte-tp7xxi"&&(O.textContent=ct),We=i(e),f(ee.$$.fragment,e),He=i(e),te=r(e,"P",{"data-svelte-h":!0}),m(te)!=="svelte-1h2v14z"&&(te.textContent=ht),Ve=i(e),f(le.$$.fragment,e),Xe=i(e),ae=r(e,"P",{"data-svelte-h":!0}),m(ae)!=="svelte-j2a68t"&&(ae.textContent=gt),Qe=i(e),f(ne.$$.fragment,e),Ye=i(e),f(v.$$.fragment,e),Se=i(e),f(ie.$$.fragment,e),Pe=i(e),se=r(e,"UL",{"data-svelte-h":!0}),m(se)!=="svelte-1rurwvs"&&(se.innerHTML=Mt),ze=i(e),oe=r(e,"P",{}),bt(oe).forEach(l),this.h()},h(){Jt(s,"name","hf:doc:metadata"),Jt(s,"content",Et)},m(e,t){Zt(document.head,s),a(e,M,t),a(e,p,t),a(e,T,t),u($,e,t),a(e,me,t),u(U,e,t),a(e,pe,t),a(e,j,t),a(e,de,t),u(R,e,t),a(e,fe,t),a(e,L,t),a(e,ue,t),a(e,Z,t),a(e,ce,t),u(C,e,t),a(e,he,t),a(e,k,t),a(e,ge,t),u(_,e,t),a(e,Me,t),a(e,A,t),a(e,ye,t),a(e,G,t),a(e,we,t),u(B,e,t),a(e,Te,t),a(e,I,t),a(e,be,t),u(E,e,t),a(e,Je,t),a(e,F,t),a(e,ve,t),a(e,x,t),a(e,$e,t),a(e,W,t),a(e,Ue,t),u(H,e,t),a(e,je,t),a(e,V,t),a(e,Re,t),a(e,X,t),a(e,Le,t),u(b,e,t),a(e,Ze,t),u(Q,e,t),a(e,Ce,t),a(e,Y,t),a(e,ke,t),u(S,e,t),a(e,_e,t),a(e,P,t),a(e,Ae,t),a(e,z,t),a(e,Ge,t),u(q,e,t),a(e,Be,t),u(J,e,t),a(e,Ie,t),u(N,e,t),a(e,Ee,t),a(e,D,t),a(e,Fe,t),a(e,K,t),a(e,xe,t),a(e,O,t),a(e,We,t),u(ee,e,t),a(e,He,t),a(e,te,t),a(e,Ve,t),u(le,e,t),a(e,Xe,t),a(e,ae,t),a(e,Qe,t),u(ne,e,t),a(e,Ye,t),u(v,e,t),a(e,Se,t),u(ie,e,t),a(e,Pe,t),a(e,se,t),a(e,ze,t),a(e,oe,t),qe=!0},p(e,[t]){const yt={};t&2&&(yt.$$scope={dirty:t,ctx:e}),b.$set(yt);const wt={};t&2&&(wt.$$scope={dirty:t,ctx:e}),J.$set(wt);const Tt={};t&2&&(Tt.$$scope={dirty:t,ctx:e}),v.$set(Tt)},i(e){qe||(c($.$$.fragment,e),c(U.$$.fragment,e),c(R.$$.fragment,e),c(C.$$.fragment,e),c(_.$$.fragment,e),c(B.$$.fragment,e),c(E.$$.fragment,e),c(H.$$.fragment,e),c(b.$$.fragment,e),c(Q.$$.fragment,e),c(S.$$.fragment,e),c(q.$$.fragment,e),c(J.$$.fragment,e),c(N.$$.fragment,e),c(ee.$$.fragment,e),c(le.$$.fragment,e),c(ne.$$.fragment,e),c(v.$$.fragment,e),c(ie.$$.fragment,e),qe=!0)},o(e){h($.$$.fragment,e),h(U.$$.fragment,e),h(R.$$.fragment,e),h(C.$$.fragment,e),h(_.$$.fragment,e),h(B.$$.fragment,e),h(E.$$.fragment,e),h(H.$$.fragment,e),h(b.$$.fragment,e),h(Q.$$.fragment,e),h(S.$$.fragment,e),h(q.$$.fragment,e),h(J.$$.fragment,e),h(N.$$.fragment,e),h(ee.$$.fragment,e),h(le.$$.fragment,e),h(ne.$$.fragment,e),h(v.$$.fragment,e),h(ie.$$.fragment,e),qe=!1},d(e){e&&(l(M),l(p),l(T),l(me),l(pe),l(j),l(de),l(fe),l(L),l(ue),l(Z),l(ce),l(he),l(k),l(ge),l(Me),l(A),l(ye),l(G),l(we),l(Te),l(I),l(be),l(Je),l(F),l(ve),l(x),l($e),l(W),l(Ue),l(je),l(V),l(Re),l(X),l(Le),l(Ze),l(Ce),l(Y),l(ke),l(_e),l(P),l(Ae),l(z),l(Ge),l(Be),l(Ie),l(Ee),l(D),l(Fe),l(K),l(xe),l(O),l(We),l(He),l(te),l(Ve),l(Xe),l(ae),l(Qe),l(Ye),l(Se),l(Pe),l(se),l(ze),l(oe)),l(s),g($,e),g(U,e),g(R,e),g(C,e),g(_,e),g(B,e),g(E,e),g(H,e),g(b,e),g(Q,e),g(S,e),g(q,e),g(J,e),g(N,e),g(ee,e),g(le,e),g(ne,e),g(v,e),g(ie,e)}}}const Et='{"title":"LoRA (Low-Rank Adaptation)","local":"lora-low-rank-adaptation","sections":[{"title":"Understanding LoRA","local":"understanding-lora","sections":[],"depth":2},{"title":"Key advantages of LoRA","local":"key-advantages-of-lora","sections":[],"depth":2},{"title":"Loading LoRA Adapters with PEFT","local":"loading-lora-adapters-with-peft","sections":[],"depth":2},{"title":"Fine-tune LLM using trl and the SFTTrainer with LoRA","local":"fine-tune-llm-using-trl-and-the-sfttrainer-with-lora","sections":[],"depth":2},{"title":"LoRA Configuration","local":"lora-configuration","sections":[],"depth":2},{"title":"Using TRL with PEFT","local":"using-trl-with-peft","sections":[],"depth":2},{"title":"Merging LoRA Adapters","local":"merging-lora-adapters","sections":[],"depth":2},{"title":"Merging Implementation","local":"merging-implementation","sections":[],"depth":2}],"depth":1}';function Ft(w){return Ut(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Yt extends jt{constructor(s){super(),Rt(this,s,Ft,It,$t,{})}}export{Yt as component};
