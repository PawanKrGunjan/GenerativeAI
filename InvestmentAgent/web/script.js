class TrishulTrader {

constructor(){

this.chat = document.getElementById("chat");
this.status = document.getElementById("statusText");

this.sessionId = localStorage.getItem("session_id") || crypto.randomUUID();
localStorage.setItem("session_id", this.sessionId);

this.niftyChart = null;
this.sensexChart = null;

this.portfolio = [];

this.bindEvents();
this.initTheme();
this.initPortfolio();

this.loadPortfolio();
this.loadMarketCharts();

setInterval(()=>this.loadMarketCharts(),60000);
setInterval(()=>this.updatePortfolioPrices(),300000);

this.welcome();

}







/* ---------------- EVENTS ---------------- */

bindEvents(){

const sendBtn = document.getElementById("sendButton");
const input = document.getElementById("messageInput");
const fileInput = document.getElementById("fileInput");

sendBtn?.addEventListener("click",()=>this.send());

input?.addEventListener("keydown",(e)=>{
if(e.key==="Enter" && !e.shiftKey){
e.preventDefault();
this.send();
}
});

fileInput?.addEventListener("change",(e)=>{
const file = e.target.files?.[0];
if(file) this.uploadPortfolio(file);
e.target.value="";
});

}







/* ---------------- CHAT ---------------- */

scrollBottom(){
requestAnimationFrame(()=>{
this.chat.scrollTop = this.chat.scrollHeight;
});
}

addMessage(text,type){

const div = document.createElement("div");
div.className = `message ${type}`;

div.innerHTML = `
<div class="avatar">
<i class="fas ${type==="ai"?"fa-robot":"fa-user"}"></i>
</div>
<div>
<div class="bubble">${text}</div>
<div class="message-time">${new Date().toLocaleTimeString()}</div>
</div>
`;

this.chat.appendChild(div);
this.scrollBottom();

}

formatAI(text){

if(!text) return "";

text = text
.replace(/Stock-Specific Advice/g,"## 📊 Stock Advice")
.replace(/Signal:/g,"**Signal:**")
.replace(/Confidence:/g,"**Confidence:**")
.replace(/Pointwise Reasoning:/g,"### Reasoning");

return marked.parse(text);

}

typeAI(text){

if(!text){
this.addMessage("No response received.","ai");
return;
}

const div = document.createElement("div");
div.className="message ai";

div.innerHTML=`
<div class="avatar"><i class="fas fa-robot"></i></div>
<div><div class="bubble"></div></div>
`;

const bubble = div.querySelector(".bubble");

this.chat.appendChild(div);

let i=0;

const type=()=>{

if(i>=text.length){
bubble.innerHTML = this.formatAI(text);
this.scrollBottom();
return;
}

bubble.textContent += text[i];
i++;

this.scrollBottom();
setTimeout(type,12);

};

type();

}







/* ---------------- SEND MESSAGE ---------------- */

async send(){

const input = document.getElementById("messageInput");
const msg = input.value.trim();

if(!msg) return;

this.addMessage(msg,"user");
input.value="";

this.status.innerText="Thinking...";

const form = new FormData();
form.append("message",msg);
form.append("session_id",this.sessionId);

try{

const res = await fetch("/chat/",{
method:"POST",
body:form
});

if(!res.ok) throw new Error();

const data = await res.json();

this.typeAI(data.answer);
this.status.innerText="Ready";

}catch(err){

console.error(err);
this.addMessage("❌ Server error.","ai");
this.status.innerText="Error";

}

}







/* ---------------- FILE UPLOAD ---------------- */

async uploadPortfolio(file){

this.addMessage(`📁 Uploading ${file.name}`,"user");
this.status.innerText="Uploading...";

const form = new FormData();
form.append("message","Analyze my portfolio");
form.append("session_id",this.sessionId);
form.append("file",file);

try{

const res = await fetch("/chat/",{
method:"POST",
body:form
});

if(!res.ok) throw new Error();

const data = await res.json();

this.typeAI(data.answer);
this.status.innerText="Ready";

}catch(err){

console.error(err);
this.addMessage("❌ Upload failed.","ai");
this.status.innerText="Error";

}

}







/* ---------------- PORTFOLIO ---------------- */

initPortfolio(){

const btn = document.getElementById("addStockBtn");

btn?.addEventListener("click",()=>{

const symbolInput = document.getElementById("symbolInput");
const qtyInput = document.getElementById("qtyInput");

const symbol = symbolInput.value.trim().toUpperCase();
const qty = Number(qtyInput.value);

if(!symbol || !qty || qty<=0){
this.addMessage("Enter valid symbol & quantity.","ai");
return;
}

this.portfolio.push({symbol,qty});

symbolInput.value="";
qtyInput.value="";

this.renderPortfolio();
this.savePortfolio();

});

}

async loadPortfolio(){

try{

const res = await fetch("/portfolio/");
if(!res.ok) return;

const data = await res.json();

this.portfolio = data?.portfolio || [];

if(this.portfolio.length){
this.renderPortfolio();
}

}catch(err){
console.error(err);
}

}

async savePortfolio(){

try{

await fetch("/portfolio/save",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify({portfolio:this.portfolio})
});

}catch(err){
console.error(err);
}

}

renderPortfolio(){

const tbody = document.querySelector("#portfolioTable tbody");
if(!tbody) return;

tbody.innerHTML="";

this.portfolio.forEach((s,i)=>{

const row = document.createElement("tr");

row.innerHTML=`
<td>${s.symbol}</td>
<td>${s.qty}</td>
<td id="price-${i}">Loading...</td>
<td id="value-${i}">-</td>
`;

tbody.appendChild(row);

});

this.updatePortfolioPrices();

}

async updatePortfolioPrices(){

if(!this.portfolio.length) return;

await Promise.all(

this.portfolio.map(async(stock,i)=>{

const priceEl = document.getElementById(`price-${i}`);
const valueEl = document.getElementById(`value-${i}`);

if(!priceEl || !valueEl) return;

try{

const symbol = stock.symbol.endsWith(".NS")
? stock.symbol
: `${stock.symbol}.NS`;

const res = await fetch(`/market/quote/${symbol}`);
const data = await res.json();

if(data.status!=="success") throw new Error();

const price = Number(data.data?.price);
const prev = data.data?.previous_close ?? price;

if(!price || isNaN(price)){
priceEl.textContent="N/A";
valueEl.textContent="—";
return;
}

const total = price * stock.qty;

const arrow = price>prev ? "▲" : price<prev ? "▼" : "→";
const color = price>prev ? "#22c55e" : price<prev ? "#ef4444" : "#64748b";

priceEl.innerHTML=`<span style="color:${color}">₹${price.toFixed(2)} ${arrow}</span>`;
valueEl.textContent=`₹${total.toFixed(2)}`;

}catch(err){

priceEl.textContent="Error";
valueEl.textContent="—";

}

})

);

this.updatePortfolioTotal();

}

updatePortfolioTotal(){

let total=0;

this.portfolio.forEach((s,i)=>{

const priceEl = document.getElementById(`price-${i}`);
if(!priceEl) return;

const price = parseFloat(
priceEl.textContent.replace(/[₹,▲▼→]/g,"")
);

if(!isNaN(price)){
total += price*s.qty;
}

});

const el = document.getElementById("portfolioTotalValue");

if(el){
el.textContent = "₹"+total.toLocaleString("en-IN",{maximumFractionDigits:2});
}

}







/* ---------------- MARKET ---------------- */

updateIndexDisplay(name,prices){

if(!prices.length) return;

const current = prices[prices.length-1];
const prev = prices[prices.length-2] ?? current;

const diff = current-prev;
const pct = (diff/prev)*100;

const arrow = diff>0 ? "▲" : diff<0 ? "▼" : "→";
const color = diff>0 ? "#22c55e" : diff<0 ? "#ef4444" : "#64748b";

const el = document.getElementById(`${name}Value`);
if(!el) return;

el.innerHTML = `
<span style="color:${color}">
${arrow} ${current.toLocaleString("en-IN")} (${pct.toFixed(2)}%)
</span>
`;

}

async loadMarketCharts(){

try{

const res = await fetch("/market/indices");
const data = await res.json();

if(data.status!=="success") return;

const labels = data.nifty.map(d=>d.date);
const nifty = data.nifty.map(d=>d.close);
const sensex = data.sensex.map(d=>d.close);

this.updateIndexDisplay("nifty",nifty);
this.updateIndexDisplay("sensex",sensex);

const ctx1=document.getElementById("niftyChart")?.getContext("2d");
const ctx2=document.getElementById("sensexChart")?.getContext("2d");

if(!ctx1 || !ctx2) return;

this.niftyChart?.destroy();
this.sensexChart?.destroy();

this.niftyChart = new Chart(ctx1,{
type:"line",
data:{labels,datasets:[{data:nifty,borderColor:"#22c55e",tension:.4,pointRadius:0}]},
options:{plugins:{legend:{display:false}}}
});

this.sensexChart = new Chart(ctx2,{
type:"line",
data:{labels,datasets:[{data:sensex,borderColor:"#ef4444",tension:.4,pointRadius:0}]},
options:{plugins:{legend:{display:false}}}
});

}catch(err){
console.error(err);
}

}







/* ---------------- THEME ---------------- */

initTheme(){

const btn=document.getElementById("themeToggle");
if(!btn) return;

let theme = localStorage.getItem("theme") || "dark";

document.body.classList.add(`theme-${theme}`);

btn.innerHTML = theme==="dark"
? '<i class="fas fa-sun"></i>'
: '<i class="fas fa-moon"></i>';

btn.onclick=()=>{

const dark=document.body.classList.contains("theme-dark");

document.body.classList.toggle("theme-dark");
document.body.classList.toggle("theme-light");

const newTheme = dark ? "light":"dark";

localStorage.setItem("theme",newTheme);

btn.innerHTML=newTheme==="dark"
? '<i class="fas fa-sun"></i>'
: '<i class="fas fa-moon"></i>';

};

}







/* ---------------- WELCOME ---------------- */

welcome(){

setTimeout(()=>{

this.typeAI(`🕉️ Namah Shivaya!

I am **Trishul Trader** — your AI trading assistant.

Ask me about:

• Indian stocks (RELIANCE, TCS, HAL)  
• NIFTY / SENSEX trends  
• Portfolio analysis  
• Upload Excel / CSV portfolio

Happy trading! 📈`);

},800);

}

}

document.addEventListener("DOMContentLoaded",()=>{
new TrishulTrader();
});