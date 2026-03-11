class TrishulTrader {

constructor(){

this.chat = document.getElementById("chat")
this.status = document.getElementById("statusText")

this.niftyChart = null
this.sensexChart = null
this.portfolio = []
this.initPortfolio()

this.bindEvents()
this.initThemeToggle()

this.loadMarketCharts()

setInterval(()=>{
this.loadMarketCharts()
},60000)

this.welcome()

}

/* ------------------ EVENTS ------------------ */

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

/* portfolio upload */

document.getElementById("fileInput")
.addEventListener("change",(e)=>{

const file = e.target.files[0]
if(!file) return

this.uploadPortfolio(file)

})

}

/* ------------------ FILE UPLOAD ------------------ */

async uploadPortfolio(file){

this.addMessage(`📁 Uploading portfolio: ${file.name}`, "user")

this.status.innerText = "Uploading..."

const form = new FormData()
form.append("message","Analyze my portfolio")
form.append("file", file)

try{

const res = await fetch("/chat/",{
method:"POST",
body:form
})

const data = await res.json()

this.typeAI(data.answer)

this.renderPortfolio(file.name)

this.status.innerText = "Ready"

}catch(err){

this.addMessage("❌ Upload failed","ai")
this.status.innerText = "Error"

}

}

/* ------------------ RENDER PORTFOLIO ------------------ */

renderPortfolio(fileName){

const panel = document.getElementById("portfolioList")

panel.innerHTML = `
<div class="portfolio-item">
<strong>${fileName}</strong>
<br>
<span>Portfolio uploaded</span>
</div>
`

}

initPortfolio(){

document
.getElementById("addStockBtn")
.addEventListener("click",()=>{

const symbol=document.getElementById("symbolInput").value.toUpperCase()
const qty=parseFloat(document.getElementById("qtyInput").value)

if(!symbol || !qty) return

this.portfolio.push({
symbol:symbol,
qty:qty
})

this.renderPortfolio()

})

/* refresh every 30 min */

setInterval(()=>{
this.updatePortfolioPrices()
},1800000)

}

/* ------------------ THEME TOGGLE ------------------ */

initThemeToggle(){

const btn = document.getElementById("themeToggle")

const saved = localStorage.getItem("theme") || "dark"

document.body.classList.remove("theme-dark","theme-light")
document.body.classList.add(`theme-${saved}`)

btn.innerHTML = saved==="dark"
? '<i class="fas fa-sun"></i>'
: '<i class="fas fa-moon"></i>'

btn.addEventListener("click",()=>{

const isDark = document.body.classList.contains("theme-dark")

document.body.classList.toggle("theme-dark")
document.body.classList.toggle("theme-light")

const newTheme = isDark ? "light" : "dark"

localStorage.setItem("theme",newTheme)

btn.innerHTML = newTheme==="dark"
? '<i class="fas fa-sun"></i>'
: '<i class="fas fa-moon"></i>'

})

}

/* ------------------ CHAT ------------------ */

scrollBottom(){
this.chat.scrollTop = this.chat.scrollHeight
}

addMessage(text,type){

const div = document.createElement("div")
div.className = `message ${type}`

const avatar = `
<div class="avatar">
<i class="fas ${type==="ai"?"fa-robot":"fa-user"}"></i>
</div>
`

div.innerHTML = `
${avatar}
<div>
<div class="bubble">${text}</div>
<div class="message-time">${new Date().toLocaleTimeString()}</div>
</div>
`

this.chat.appendChild(div)

this.scrollBottom()

}

/* ------------------ AI TYPING ------------------ */

typeAI(text){

const div = document.createElement("div")
div.className = "message ai"

div.innerHTML = `
<div class="avatar">
<i class="fas fa-robot"></i>
</div>
<div>
<div class="bubble"></div>
</div>
`

const bubble = div.querySelector(".bubble")

this.chat.appendChild(div)

let i = 0

const typing = ()=>{

if(i < text.length){

bubble.textContent += text[i]
i++

this.scrollBottom()

setTimeout(typing,15)

}

}

typing()

}

/* ------------------ SEND MESSAGE ------------------ */

async send(){

const input = document.getElementById("messageInput")

const msg = input.value.trim()

if(!msg) return

this.addMessage(msg,"user")

input.value = ""

this.status.innerText = "Thinking..."

try{

const form = new FormData()
form.append("message",msg)

const res = await fetch("/chat/",{
method:"POST",
body:form
})

const data = await res.json()

this.typeAI(data.answer)

this.status.innerText = "Ready"

}catch(e){

this.addMessage("❌ Server connection error","ai")
this.status.innerText = "Error"

}

}

/* ------------------ MARKET TREND ------------------ */

getTrend(prices){

if(prices.length < 2) return "neutral"

const last = prices[prices.length-1]
const prev = prices[prices.length-2]

return last > prev ? "up" : "down"

}

/* ------------------ APPLY MARKET THEME ------------------ */

applyTheme(trend){

document.body.classList.remove("market-up","market-down")

if(trend==="up"){
document.body.classList.add("market-up")
}else{
document.body.classList.add("market-down")
}

}

/* ------------------ LOAD MARKET CHARTS ------------------ */

async loadMarketCharts(){

try{

const res = await fetch("/market/indices")
const data = await res.json()

if(data.status !== "success") return

const labels = data.nifty.map(d=>d.date)

const niftyPrices = data.nifty.map(d=>d.close)
const sensexPrices = data.sensex.map(d=>d.close)

const trend = this.getTrend(niftyPrices)

const arrow = trend==="up" ? "▲" : "▼"
const direction = trend==="up" ? "UP" : "DOWN"
const color = trend==="up" ? "#22c55e" : "#ef4444"

this.applyTheme(trend)

/* latest values */

const niftyLast = niftyPrices[niftyPrices.length-1]
const niftyPrev = niftyPrices[niftyPrices.length-2]

const sensexLast = sensexPrices[sensexPrices.length-1]
const sensexPrev = sensexPrices[sensexPrices.length-2]

const niftyChange = ((niftyLast-niftyPrev)/niftyPrev)*100
const sensexChange = ((sensexLast-sensexPrev)/sensexPrev)*100

/* update labels */

document.querySelector(".nifty-label").innerHTML =
`NIFTY 50 <span style="color:${color}">
${arrow} ${direction} ${niftyChange.toFixed(2)}%
</span> • ${niftyLast.toFixed(2)}`

document.querySelector(".sensex-label").innerHTML =
`SENSEX <span style="color:${color}">
${arrow} ${direction} ${sensexChange.toFixed(2)}%
</span> • ${sensexLast.toFixed(2)}`

/* destroy old charts */

if(this.niftyChart) this.niftyChart.destroy()
if(this.sensexChart) this.sensexChart.destroy()

const niftyCanvas = document.getElementById("niftyChart")
const sensexCanvas = document.getElementById("sensexChart")

/* gradients */

const g1 = niftyCanvas.getContext("2d").createLinearGradient(0,0,0,220)
const g2 = sensexCanvas.getContext("2d").createLinearGradient(0,0,0,220)

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

/* NIFTY */

this.niftyChart = new Chart(niftyCanvas,{
type:"line",
data:{
labels:labels,
datasets:[{
data:niftyPrices,
borderColor:color,
backgroundColor:g1,
fill:true,
tension:0.4,
pointRadius:0,
borderWidth:2
}]
},
options:{
responsive:true,
maintainAspectRatio:false,
plugins:{legend:{display:false}},
scales:{
x:{grid:{color:"rgba(150,150,150,0.15)"},ticks:{display:false}},
y:{grid:{color:"rgba(150,150,150,0.15)"},ticks:{display:false}}
}
}
})

/* SENSEX */

this.sensexChart = new Chart(sensexCanvas,{
type:"line",
data:{
labels:labels,
datasets:[{
data:sensexPrices,
borderColor:color,
backgroundColor:g2,
fill:true,
tension:0.4,
pointRadius:0,
borderWidth:2
}]
},
options:{
responsive:true,
maintainAspectRatio:false,
plugins:{legend:{display:false}},
scales:{
x:{grid:{color:"rgba(150,150,150,0.15)"},ticks:{display:false}},
y:{grid:{color:"rgba(150,150,150,0.15)"},ticks:{display:false}}
}
}
})

}catch(err){

console.error("Chart error",err)

}

}

/* ------------------ WELCOME ------------------ */

welcome(){

setTimeout(()=>{

this.typeAI(
"🕉️ Namah Shivaya!\n\nI am your AI trading assistant.\nAsk me about stocks, market trends or upload portfolio."
)

},700)

}

}

/* ------------------ START APP ------------------ */

document.addEventListener("DOMContentLoaded",()=>{
new TrishulTrader()
})