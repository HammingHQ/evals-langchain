var te=Object.create;var A=Object.defineProperty;var re=Object.getOwnPropertyDescriptor;var ne=Object.getOwnPropertyNames;var ae=Object.getPrototypeOf,oe=Object.prototype.hasOwnProperty;var se=(e,t,n)=>t in e?A(e,t,{enumerable:!0,configurable:!0,writable:!0,value:n}):e[t]=n;var N=(e,t)=>()=>(t||e((t={exports:{}}).exports,t),t.exports);var ie=(e,t,n,r)=>{if(t&&typeof t=="object"||typeof t=="function")for(let a of ne(t))!oe.call(e,a)&&a!==n&&A(e,a,{get:()=>t[a],enumerable:!(r=re(t,a))||r.enumerable});return e};var D=(e,t,n)=>(n=e!=null?te(ae(e)):{},ie(t||!e||!e.__esModule?A(n,"default",{value:e,enumerable:!0}):n,e));var f=(e,t,n)=>se(e,typeof t!="symbol"?t+"":t,n);var B=N((Ie,J)=>{"use strict";J.exports=function(e,t){if(typeof e!="string")throw new TypeError("Expected a string");return t=typeof t>"u"?"_":t,e.replace(/([a-z\d])([A-Z])/g,"$1"+t+"$2").replace(/([A-Z]+)([A-Z][a-z\d]+)/g,"$1"+t+"$2").toLowerCase()}});var K=N((je,j)=>{"use strict";var fe=/[\p{Lu}]/u,me=/[\p{Ll}]/u,z=/^[\p{Lu}](?![\p{Lu}])/gu,$=/([\p{Alpha}\p{N}_]|$)/u,F=/[_.\- ]+/,pe=new RegExp("^"+F.source),H=new RegExp(F.source+$.source,"gu"),G=new RegExp("\\d+"+$.source,"gu"),ge=(e,t,n)=>{let r=!1,a=!1,o=!1;for(let i=0;i<e.length;i++){let s=e[i];r&&fe.test(s)?(e=e.slice(0,i)+"-"+e.slice(i),r=!1,o=a,a=!0,i++):a&&o&&me.test(s)?(e=e.slice(0,i-1)+"-"+e.slice(i-1),o=a,a=!1,r=!0):(r=t(s)===s&&n(s)!==s,o=a,a=n(s)===s&&t(s)!==s)}return e},_e=(e,t)=>(z.lastIndex=0,e.replace(z,n=>t(n))),he=(e,t)=>(H.lastIndex=0,G.lastIndex=0,e.replace(H,(n,r)=>t(r)).replace(G,n=>t(n))),V=(e,t)=>{if(!(typeof e=="string"||Array.isArray(e)))throw new TypeError("Expected the input to be `string | string[]`");if(t={pascalCase:!1,preserveConsecutiveUppercase:!1,...t},Array.isArray(e)?e=e.map(o=>o.trim()).filter(o=>o.length).join("-"):e=e.trim(),e.length===0)return"";let n=t.locale===!1?o=>o.toLowerCase():o=>o.toLocaleLowerCase(t.locale),r=t.locale===!1?o=>o.toUpperCase():o=>o.toLocaleUpperCase(t.locale);return e.length===1?t.pascalCase?r(e):n(e):(e!==n(e)&&(e=ge(e,n,r)),e=e.replace(pe,""),t.preserveConsecutiveUppercase?e=_e(e,n):e=n(e),t.pascalCase&&(e=r(e.charAt(0))+e.slice(1)),he(e,r))};j.exports=V;j.exports.default=V});import le from"crypto";var O=new Uint8Array(256),M=O.length;function T(){return M>O.length-16&&(le.randomFillSync(O),M=0),O.slice(M,M+=16)}var u=[];for(let e=0;e<256;++e)u.push((e+256).toString(16).slice(1));function U(e,t=0){return u[e[t+0]]+u[e[t+1]]+u[e[t+2]]+u[e[t+3]]+"-"+u[e[t+4]]+u[e[t+5]]+"-"+u[e[t+6]]+u[e[t+7]]+"-"+u[e[t+8]]+u[e[t+9]]+"-"+u[e[t+10]]+u[e[t+11]]+u[e[t+12]]+u[e[t+13]]+u[e[t+14]]+u[e[t+15]]}import ce from"crypto";var R={randomUUID:ce.randomUUID};function ue(e,t,n){if(R.randomUUID&&!t&&!e)return R.randomUUID();e=e||{};let r=e.random||(e.rng||T)();if(r[6]=r[6]&15|64,r[8]=r[8]&63|128,t){n=n||0;for(let a=0;a<16;++a)t[n+a]=r[a];return t}return U(r)}var I=ue;var W=D(B(),1),ye=D(K(),1);function q(e,t){return t?.[e]||(0,W.default)(e)}function Q(e,t,n){let r={};for(let a in e)Object.hasOwn(e,a)&&(r[t(a,n)]=e[a]);return r}function Z(e){return Array.isArray(e)?[...e]:{...e}}function we(e,t){let n=Z(e);for(let[r,a]of Object.entries(t)){let[o,...i]=r.split(".").reverse(),s=n;for(let c of i.reverse()){if(s[c]===void 0)break;s[c]=Z(s[c]),s=s[c]}s[o]!==void 0&&(s[o]={lc:1,type:"secret",id:[a]})}return n}function E(e){let t=Object.getPrototypeOf(e);return typeof e.lc_name=="function"&&(typeof t.lc_name!="function"||e.lc_name()!==t.lc_name())?e.lc_name():e.name}var p=class e{static lc_name(){return this.name}get lc_id(){return[...this.lc_namespace,E(this.constructor)]}get lc_secrets(){}get lc_attributes(){}get lc_aliases(){}constructor(t,...n){Object.defineProperty(this,"lc_serializable",{enumerable:!0,configurable:!0,writable:!0,value:!1}),Object.defineProperty(this,"lc_kwargs",{enumerable:!0,configurable:!0,writable:!0,value:void 0}),this.lc_kwargs=t||{}}toJSON(){if(!this.lc_serializable)return this.toJSONNotImplemented();if(this.lc_kwargs instanceof e||typeof this.lc_kwargs!="object"||Array.isArray(this.lc_kwargs))return this.toJSONNotImplemented();let t={},n={},r=Object.keys(this.lc_kwargs).reduce((a,o)=>(a[o]=o in this?this[o]:this.lc_kwargs[o],a),{});for(let a=Object.getPrototypeOf(this);a;a=Object.getPrototypeOf(a))Object.assign(t,Reflect.get(a,"lc_aliases",this)),Object.assign(n,Reflect.get(a,"lc_secrets",this)),Object.assign(r,Reflect.get(a,"lc_attributes",this));return Object.keys(n).forEach(a=>{let o=this,i=r,[s,...c]=a.split(".").reverse();for(let l of c.reverse()){if(!(l in o)||o[l]===void 0)return;(!(l in i)||i[l]===void 0)&&(typeof o[l]=="object"&&o[l]!=null?i[l]={}:Array.isArray(o[l])&&(i[l]=[])),o=o[l],i=i[l]}s in o&&o[s]!==void 0&&(i[s]=i[s]||o[s])}),{lc:1,type:"constructor",id:this.lc_id,kwargs:Q(Object.keys(n).length?we(r,n):r,q,t)}}toJSONNotImplemented(){return{lc:1,type:"not_implemented",id:this.lc_id}}};function X(e){try{return typeof process<"u"?process.env?.[e]:void 0}catch{return}}var L=class{},S=class e extends L{get lc_namespace(){return["langchain_core","callbacks",this.name]}get lc_secrets(){}get lc_attributes(){}get lc_aliases(){}static lc_name(){return this.name}get lc_id(){return[...this.lc_namespace,E(this.constructor)]}constructor(t){super(),Object.defineProperty(this,"lc_serializable",{enumerable:!0,configurable:!0,writable:!0,value:!1}),Object.defineProperty(this,"lc_kwargs",{enumerable:!0,configurable:!0,writable:!0,value:void 0}),Object.defineProperty(this,"ignoreLLM",{enumerable:!0,configurable:!0,writable:!0,value:!1}),Object.defineProperty(this,"ignoreChain",{enumerable:!0,configurable:!0,writable:!0,value:!1}),Object.defineProperty(this,"ignoreAgent",{enumerable:!0,configurable:!0,writable:!0,value:!1}),Object.defineProperty(this,"ignoreRetriever",{enumerable:!0,configurable:!0,writable:!0,value:!1}),Object.defineProperty(this,"raiseError",{enumerable:!0,configurable:!0,writable:!0,value:!1}),Object.defineProperty(this,"awaitHandlers",{enumerable:!0,configurable:!0,writable:!0,value:X("LANGCHAIN_CALLBACKS_BACKGROUND")!=="true"}),this.lc_kwargs=t||{},t&&(this.ignoreLLM=t.ignoreLLM??this.ignoreLLM,this.ignoreChain=t.ignoreChain??this.ignoreChain,this.ignoreAgent=t.ignoreAgent??this.ignoreAgent,this.ignoreRetriever=t.ignoreRetriever??this.ignoreRetriever,this.raiseError=t.raiseError??this.raiseError,this.awaitHandlers=this.raiseError||(t._awaitHandler??this.awaitHandlers))}copy(){return new this.constructor(this)}toJSON(){return p.prototype.toJSON.call(this)}toJSONNotImplemented(){return p.prototype.toJSONNotImplemented.call(this)}static fromMethods(t){class n extends e{constructor(){super(),Object.defineProperty(this,"name",{enumerable:!0,configurable:!0,writable:!0,value:I()}),Object.assign(this,t)}}return new n}};var d=class extends p{get lc_aliases(){return{additional_kwargs:"additional_kwargs",response_metadata:"response_metadata"}}get text(){return typeof this.content=="string"?this.content:""}constructor(t,n){typeof t=="string"&&(t={content:t,additional_kwargs:n,response_metadata:{}}),t.additional_kwargs||(t.additional_kwargs={}),t.response_metadata||(t.response_metadata={}),super(t),Object.defineProperty(this,"lc_namespace",{enumerable:!0,configurable:!0,writable:!0,value:["langchain_core","messages"]}),Object.defineProperty(this,"lc_serializable",{enumerable:!0,configurable:!0,writable:!0,value:!0}),Object.defineProperty(this,"content",{enumerable:!0,configurable:!0,writable:!0,value:void 0}),Object.defineProperty(this,"name",{enumerable:!0,configurable:!0,writable:!0,value:void 0}),Object.defineProperty(this,"additional_kwargs",{enumerable:!0,configurable:!0,writable:!0,value:void 0}),Object.defineProperty(this,"response_metadata",{enumerable:!0,configurable:!0,writable:!0,value:void 0}),this.name=t.name,this.content=t.content,this.additional_kwargs=t.additional_kwargs,this.response_metadata=t.response_metadata}toDict(){return{type:this._getType(),data:this.toJSON().kwargs}}};var g=class extends d{static lc_name(){return"ToolMessage"}get lc_aliases(){return{tool_call_id:"tool_call_id"}}constructor(t,n,r){typeof t=="string"&&(t={content:t,name:r,tool_call_id:n}),super(t),Object.defineProperty(this,"tool_call_id",{enumerable:!0,configurable:!0,writable:!0,value:void 0}),this.tool_call_id=t.tool_call_id}_getType(){return"tool"}static isInstance(t){return t._getType()==="tool"}};function Y(e){let t=[],n=[];for(let r of e)if(r.function){let a=r.function.name;try{let o=JSON.parse(r.function.arguments),i={name:a||"",args:o||{},id:r.id};t.push(i)}catch{n.push({name:a,args:r.function.arguments,id:r.id,error:"Malformed args."})}}else continue;return[t,n]}var w=class extends d{get lc_aliases(){return{...super.lc_aliases,tool_calls:"tool_calls",invalid_tool_calls:"invalid_tool_calls"}}constructor(t,n){let r;if(typeof t=="string")r={content:t,tool_calls:[],invalid_tool_calls:[],additional_kwargs:n??{}};else{r=t;let a=r.additional_kwargs?.tool_calls,o=r.tool_calls;a!=null&&a.length>0&&(o===void 0||o.length===0)&&console.warn(["New LangChain packages are available that more efficiently handle",`tool calling.

Please upgrade your packages to versions that set`,"message tool calls. e.g., `yarn add @langchain/anthropic`,","yarn add @langchain/openai`, etc."].join(" "));try{if(a!=null&&o===void 0){let[i,s]=Y(a);r.tool_calls=i??[],r.invalid_tool_calls=s??[]}else r.tool_calls=r.tool_calls??[],r.invalid_tool_calls=r.invalid_tool_calls??[]}catch{r.tool_calls=[],r.invalid_tool_calls=[]}}super(r),Object.defineProperty(this,"tool_calls",{enumerable:!0,configurable:!0,writable:!0,value:[]}),Object.defineProperty(this,"invalid_tool_calls",{enumerable:!0,configurable:!0,writable:!0,value:[]}),Object.defineProperty(this,"usage_metadata",{enumerable:!0,configurable:!0,writable:!0,value:void 0}),typeof r!="string"&&(this.tool_calls=r.tool_calls??this.tool_calls,this.invalid_tool_calls=r.invalid_tool_calls??this.invalid_tool_calls),this.usage_metadata=r.usage_metadata}static lc_name(){return"AIMessage"}_getType(){return"ai"}};var b=class e extends d{static lc_name(){return"ChatMessage"}static _chatMessageClass(){return e}constructor(t,n){typeof t=="string"&&(t={content:t,role:n}),super(t),Object.defineProperty(this,"role",{enumerable:!0,configurable:!0,writable:!0,value:void 0}),this.role=t.role}_getType(){return"generic"}static isInstance(t){return t._getType()==="generic"}};var v=class extends d{static lc_name(){return"FunctionMessage"}constructor(t,n){typeof t=="string"&&(t={content:t,name:n}),super(t)}_getType(){return"function"}};var x=class extends d{static lc_name(){return"HumanMessage"}_getType(){return"human"}};var k=class extends d{static lc_name(){return"SystemMessage"}_getType(){return"system"}};var ee=class extends S{constructor(n){super();f(this,"name","HammingCallbackHandler");f(this,"hamming");f(this,"runItems",{});f(this,"runParent",{});f(this,"runLlmInput",{});f(this,"runLlmSerialized",{});f(this,"runRetrieverQuery",{});f(this,"runRetrieverSerialized",{});this.hamming=n,this.hamming.monitoring.start()}async handleChainStart(n,r,a,o,i,s,c,l){if(console.log("HCB-1: Chain Started"),o){this.runParent[a]=o;return}let m=await this.hamming.monitoring.startItem();m.setInput(r),this.runItems[a]=m}async handleChainEnd(n,r,a){if(a){delete this.runParent[r];return}let o=this.runItems[r];if(!o){console.warn("No monitoring item found for runId: ",r);return}delete this.runItems[r];let i=JSON.parse(JSON.stringify(n));o.setOutput({response:i.kwargs?.content}),o.end()}async handleChatModelStart(n,r,a,o,i,s,c,l){let C=r.flat().map(P=>be(P));this.runLlmInput[a]=JSON.stringify(C),this.runLlmSerialized[a]=n}async handleLLMStart(n,r,a,o,i,s,c,l){this.runLlmInput[a]=JSON.stringify(r),this.runLlmSerialized[a]=n}async handleLLMEnd(n,r,a,o){let i=this.runLlmInput[r],s=this.runLlmSerialized[r],c=n.generations[n.generations.length-1],l=c[c.length-1],m={input:i,output:l.text};if(s.id.includes("openai")&&(m.metadata={provider:"openai",model:s.kwargs.model,temperature:s.kwargs.temperature}),a){let C=this._findTopParentRunId(a),P=this.runItems[C];if(!P){console.warn("No monitoring item found for runId: ",C);return}P.tracing.logGeneration(m)}else this.hamming.tracing.logGeneration(m)}async handleRetrieverStart(n,r,a,o,i,s,c){this.runRetrieverQuery[a]=r,this.runRetrieverSerialized[a]=n}async handleRetrieverEnd(n,r,a){let o=this.runRetrieverQuery[r],i=this.runRetrieverSerialized[r],s={query:o,results:n.map(c=>c.pageContent),metadata:{engine:"langchain"}};if(a){let c=this._findTopParentRunId(a),l=this.runItems[c];if(!l){console.warn("No monitoring item found for runId: ",c);return}l.tracing.logRetrieval(s)}else this.hamming.tracing.logRetrieval(s)}_findTopParentRunId(n){let r=this.runParent[n];return r?this._findTopParentRunId(r):n}};function be(e){let t;return e instanceof x?t={content:e.content,role:"user"}:e instanceof b?t={content:e.content,role:e.name}:e instanceof w?t={content:e.content,role:"assistant"}:e instanceof k?t={content:e.content,role:"system"}:e instanceof v?t={content:e.content,additional_kwargs:e.additional_kwargs,role:e.name}:e instanceof g?t={content:e.content,additional_kwargs:e.additional_kwargs,role:e.name}:e.name?t={role:e.name,content:e.content}:t={content:e.content},t}export{ee as HammingCallbackHandler};
//# sourceMappingURL=index.js.map