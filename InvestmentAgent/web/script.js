class TrishulTrader {

/*
constructor(){

this.chat = document.getElementById("chat")
this.status = document.getElementById("statusText")

this.niftyChart = null
this.sensexChart = null

this.portfolio = []

this.bindEvents()
this.initThemeToggle()
this.initPortfolio()

this.loadMarketCharts()

setInterval(()=>this.loadMarketCharts(),60000)

this.welcome()

}
*/
constructor(){

this.chat = document.getElementById("chat")
this.status = document.getElementById("statusText")

this.niftyChart = null
this.sensexChart = null

this.portfolio = []

this.bindEvents()
this.initThemeToggle()

this.initPortfolio()

this.loadPortfolio()   // ⭐ load saved portfolio

this.loadMarketCharts()

setInterval(()=>this.loadMarketCharts(),60000)

this.welcome()

}

/* ---------------- EVENTS ---------------- */

bindEvents(){

document.getElementById("sendButton")
.addEventListener("click",()=>this.send())

document.getElementById("messageInput")
.addEventListener("keypress",(e)=>{
if(e.key==="Enter"){
e.preventDefault()
this.send()
}
})

document.getElementById("fileInput")
.addEventListener("change",(e)=>{
const file = e.target.files[0]
if(file) this.uploadPortfolio(file)
})

}

/* ---------------- FILE UPLOAD ---------------- */

async uploadPortfolio(file){

this.addMessage(`📁 Uploading portfolio: ${file.name}`,"user")

this.status.innerText="Uploading..."

const form=new FormData()
form.append("message","Analyze my portfolio")
form.append("file",file)

try{

const res=await fetch("/chat/",{
method:"POST",
body:form
})

const data=await res.json()

this.typeAI(data.answer)

this.showUploadedFile(file.name)

this.status.innerText="Ready"

}catch(err){

this.addMessage("❌ Upload failed","ai")
this.status.innerText="Error"

}

}

/* ---------------- SHOW UPLOADED FILE ---------------- */

showUploadedFile(name){

const panel=document.getElementById("portfolioList")

panel.innerHTML=`

<div class="portfolio-item">
<strong>${name}</strong>
<br>
<span>Portfolio uploaded</span>
</div>
`

}

/* ---------------- PORTFOLIO SYSTEM ---------------- */
initPortfolio(){

document.getElementById("addStockBtn")
.addEventListener("click",()=>{

const symbol=document.getElementById("symbolInput").value.toUpperCase()
const qty=parseFloat(document.getElementById("qtyInput").value)

if(!symbol || !qty) return

this.portfolio.push({
symbol:symbol,
qty:qty
})

this.renderPortfolioTable()

this.savePortfolio() // ⭐ save to backend

})

/* refresh every 30 minutes */

setInterval(()=>this.updatePortfolioPrices(),1800000)

}
/* ---------------- LOAD PORTFOLIO FROM BACKEND ---------------- */

async loadPortfolio(){

try{

const res = await fetch("/portfolio/")
const data = await res.json()

this.portfolio = data.portfolio || []

if(this.portfolio.length){
this.renderPortfolioTable()
}

}catch(err){

console.error("Portfolio load failed")

}

}

/* ---------------- SAVE PORTFOLIO TO BACKEND ---------------- */

async savePortfolio(){

try{

await fetch("/portfolio/save",{

method:"POST",

headers:{
"Content-Type":"application/json"
},

body:JSON.stringify({
portfolio:this.portfolio
})

})

}catch(err){

console.error("Portfolio save failed")

}

}

/* ---------------- RENDER PORTFOLIO TABLE ---------------- */

renderPortfolioTable(){

const tbody=document.querySelector("#portfolioTable tbody")

tbody.innerHTML=""

this.portfolio.forEach((stock,i)=>{

const row=document.createElement("tr")

row.innerHTML=`

<td>${stock.symbol}</td>
<td>${stock.qty}</td>
<td id="price-${i}">Loading...</td>
<td id="value-${i}">-</td>

`

tbody.appendChild(row)

})

this.updatePortfolioPrices()

}

/* ---------------- UPDATE PORTFOLIO PRICES ---------------- */
async updatePortfolioPrices(){

for(let i=0;i<this.portfolio.length;i++){

const stock=this.portfolio[i]

try{

const res = await fetch(`/market/quote/${stock.symbol}`)
const data = await res.json()

if(data.status !== "success") throw "API error"

const price = parseFloat(data.data.last_price)

const prev = data.data.previous_close
? parseFloat(data.data.previous_close)
: price

const trend = price > prev ? "▲" : "▼"
const color = price > prev ? "#22c55e" : "#ef4444"

const total = price * stock.qty

document.getElementById(`price-${i}`).innerHTML =
`<span style="color:${color}">₹${price.toFixed(2)} ${trend}</span>`

document.getElementById(`value-${i}`).innerText =
`₹${total.toFixed(2)}`

}catch(err){

document.getElementById(`price-${i}`).innerText="Error"

}

}

}

/* ---------------- THEME ---------------- */

initThemeToggle(){

const btn=document.getElementById("themeToggle")

const saved=localStorage.getItem("theme")||"dark"

document.body.classList.remove("theme-dark","theme-light")
document.body.classList.add(`theme-${saved}`)

btn.innerHTML=saved==="dark"
?'<i class="fas fa-sun"></i>'
:'<i class="fas fa-moon"></i>'

btn.addEventListener("click",()=>{

const isDark=document.body.classList.contains("theme-dark")

document.body.classList.toggle("theme-dark")
document.body.classList.toggle("theme-light")

const newTheme=isDark?"light":"dark"

localStorage.setItem("theme",newTheme)

btn.innerHTML=newTheme==="dark"
?'<i class="fas fa-sun"></i>'
:'<i class="fas fa-moon"></i>'

})

}

/* ---------------- CHAT UI ---------------- */

scrollBottom(){
this.chat.scrollTop=this.chat.scrollHeight
}

addMessage(text,type){

const div=document.createElement("div")
div.className=`message ${type}`

div.innerHTML=`

<div class="avatar">
<i class="fas ${type==="ai"?"fa-robot":"fa-user"}"></i>
</div>
<div>
<div class="bubble">${text}</div>
<div class="message-time">${new Date().toLocaleTimeString()}</div>
</div>
`

this.chat.appendChild(div)
this.scrollBottom()

}

typeAI(text){

const div=document.createElement("div")
div.className="message ai"

div.innerHTML=`

<div class="avatar">
<i class="fas fa-robot"></i>
</div>
<div>
<div class="bubble"></div>
</div>
`

const bubble=div.querySelector(".bubble")

this.chat.appendChild(div)

let i=0

const typing=()=>{

if(i<text.length){

bubble.textContent+=text[i]
i++

this.scrollBottom()

setTimeout(typing,15)

}

}

typing()

}

/* ---------------- SEND MESSAGE ---------------- */

async send(){

const input=document.getElementById("messageInput")

const msg=input.value.trim()

if(!msg) return

this.addMessage(msg,"user")

input.value=""

this.status.innerText="Thinking..."

try{

const form=new FormData()
form.append("message",msg)

const res=await fetch("/chat/",{
method:"POST",
body:form
})

const data=await res.json()

this.typeAI(data.answer)

this.status.innerText="Ready"

}catch(err){

this.addMessage("❌ Server error","ai")
this.status.innerText="Error"

}

}

/* ---------------- MARKET DATA ---------------- */

getTrend(prices){

if(prices.length<2) return "neutral"

const last=prices[prices.length-1]
const prev=prices[prices.length-2]

return last>prev?"up":"down"

}

applyTheme(trend){

document.body.classList.remove("market-up","market-down")

document.body.classList.add(trend==="up"?"market-up":"market-down")

}

/* ---------------- LOAD CHARTS ---------------- */

async loadMarketCharts(){

try{

const res=await fetch("/market/indices")
const data=await res.json()

if(data.status!=="success") return

const labels=data.nifty.map(d=>d.date)

const niftyPrices=data.nifty.map(d=>d.close)
const sensexPrices=data.sensex.map(d=>d.close)

const trend=this.getTrend(niftyPrices)

const color=trend==="up"?"#22c55e":"#ef4444"

this.applyTheme(trend)

if(this.niftyChart) this.niftyChart.destroy()
if(this.sensexChart) this.sensexChart.destroy()

const g1=document.getElementById("niftyChart").getContext("2d").createLinearGradient(0,0,0,220)
const g2=document.getElementById("sensexChart").getContext("2d").createLinearGradient(0,0,0,220)

if(trend==="up"){
g1.addColorStop(0,"rgba(34,197,94,0.35)")
g1.addColorStop(1,"rgba(34,197,94,0.02)")
g2.addColorStop(0,"rgba(34,197,94,0.35)")
g2.addColorStop(1,"rgba(34,197,94,0.02)")
}else{
g1.addColorStop(0,"rgba(239,68,68,0.35)")
g1.addColorStop(1,"rgba(239,68,68,0.02)")
g2.addColorStop(0,"rgba(239,68,68,0.35)")
g2.addColorStop(1,"rgba(239,68,68,0.02)")
}

this.niftyChart=new Chart(document.getElementById("niftyChart"),{
type:"line",
data:{labels,datasets:[{data:niftyPrices,borderColor:color,backgroundColor:g1,fill:true,tension:0.4,pointRadius:0}]},
options:{responsive:true,plugins:{legend:{display:false}}}
})

this.sensexChart=new Chart(document.getElementById("sensexChart"),{
type:"line",
data:{labels,datasets:[{data:sensexPrices,borderColor:color,backgroundColor:g2,fill:true,tension:0.4,pointRadius:0}]},
options:{responsive:true,plugins:{legend:{display:false}}}
})

}catch(err){
console.error("Chart error",err)
}

}

/* ---------------- WELCOME ---------------- */

welcome(){

setTimeout(()=>{
this.typeAI("🕉️ Namah Shivaya!\n\nI am your AI trading assistant.\nAsk about stocks, trends, or upload your portfolio.")
},700)

}

}

document.addEventListener("DOMContentLoaded",()=>{
new TrishulTrader()
})
