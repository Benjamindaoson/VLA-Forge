"use strict";
let snapshot = null;
let selected = null;
const byId = id => document.getElementById(id);
function node(tag, text, className) { const n = document.createElement(tag); if (text !== undefined) n.textContent = text; if (className) n.className = className; return n; }
function link(key, label) { const a = node("a", label); a.href = "/api/evidence/" + encodeURIComponent(key); a.target = "_blank"; a.rel = "noopener"; return a; }
const taskName = row => row.suite === "libero_10" ? "两罐汤入篮" : "打开中间抽屉";
const evidenceLabel = {gate0:"原始失败记录",prefix:"前缀重放证据","source-audit":"纠正来源审计"};
function detail(row) {
  const area = byId("case-detail"); area.replaceChildren();
  if (!row) { area.append(node("p", "没有通过校验的案例可供展示。")); return; }
  area.append(node("span", "归档失败案例", "badge"), node("h2", taskName(row), "detail-title"), node("p", row.instruction, "caption"));
  const fields = [["任务", row.suite + " / task " + row.task_id], ["干预位置", "第 " + row.prefix + " 步，剩余 " + row.remaining_steps + " 步"], ["初态 / 环境seed", row.initial_state + " / " + row.environment_seed], ["原始结果", "未完成任务（" + row.original_steps + " 步）"], ["证据来源", row.observed_on]];
  const dl = node("dl"); for (const [k,v] of fields) dl.append(node("dt",k),node("dd",String(v))); area.append(dl);
  area.append(node("h3", "证据链")); const chain = node("ul", undefined, "evidence-chain");
  for (const [k,v,ok] of [["原始失败","已记录",true],["历史前缀重放","归档通过",true],["3090同场景重建","待执行",false],["可信纠正","尚未取得",false],["修复训练与回归","待前序门槛",false]]) { const li=node("li");li.append(node("span",k),node("span",v,"badge"+(ok?" ok":"")));chain.append(li); } area.append(chain);
  const message = row.raw_trajectory_status === "MISSING" ? "原始轨迹数组尚未迁入。视频、动作曲线与同步重放暂不可用；页面只展示已核验的归档元数据。" : "已发现轨迹文件，当前页面未重新核验它的内容。需要通过采集预检后才能用于重放。";
  area.append(node("div",message,"missing")); const sources=node("div",undefined,"sources"); for(const key of row.evidence_keys)sources.append(link(key,evidenceLabel[key])); area.append(sources);
}
function renderCases() {
  const filter=byId("task-filter").value;const rows=snapshot.incidents.filter(r=>filter==="all"||r.suite===filter);
  if(!rows.some(r=>r.case_id===selected)) selected=rows[0]?.case_id??null;
  const list=byId("case-table");list.replaceChildren();
  for(const row of rows){const b=node("button",undefined,"case-button"+(selected===row.case_id?" selected":""));b.setAttribute("aria-pressed",String(selected===row.case_id));b.append(node("strong",taskName(row)+" · 前缀 "+row.prefix),node("small",row.suite+" / init 0 / 原始任务未成功"));b.addEventListener("click",()=>{selected=row.case_id;renderCases();});list.append(b);}
  if(!rows.length)list.append(node("p","当前没有可展示的已验证案例。"));detail(rows.find(r=>r.case_id===selected));
}
function renderRuntime(){
  const area=byId("runtime-detail");area.replaceChildren();const runtime=snapshot.runtime;
  area.append(node("p",runtime.status==="MISSING"?"尚无新机验收回执。":runtime.status==="INVALID"?"验收回执无法读取，请检查源证据。":"回执状态："+runtime.status),node("p","记录时间："+(runtime.observed_utc??"未记录"),"receipt-note"));
  const names={doctor:"环境检查",linux_full_tests:"Linux / CUDA 测试",gpu_scene_smoke:"基础模型真实场景联调",gpu_training_profile:"短程训练可行性检查",collector_preflight_remote:"纠正资产预检"};
  const table=node("table",undefined,"metric-table");const head=node("tr");["检查","结果","耗时"].forEach(t=>head.append(node("th",t)));table.append(head);
  for(const step of runtime.steps){const row=node("tr");let label=step.exit_code===0?"通过":"未通过";if(step.name==="collector_preflight_remote"&&step.exit_code===2)label="缺少历史资产";row.append(node("td",names[step.name]??step.name),node("td",label),node("td",Number(step.seconds).toFixed(1)+" 秒"));table.append(row);}area.append(table);
  const assets=snapshot.transfer.data?.assets??[];if(assets.length){area.append(node("h3","已保存的迁移回执"));for(const a of assets)area.append(node("p",a.path+"："+a.files+" 个文件，"+(a.bytes/1e9).toFixed(3)+" GB","caption"));}
  area.append(node("div","基础模型烟雾运行和短程训练只验证工程链路；它们不恢复历史微调模型，也不构成策略修复效果。","missing"));const sources=node("div",undefined,"sources");for(const [key,label] of [["readiness","验收原始回执"],["transfer","迁移核验"],["preflight","缺失资产清单"]])if(snapshot.evidence.some(e=>e.key===key&&e.available))sources.append(link(key,label));area.append(sources);
}
function renderRelease(){const area=byId("release-detail");area.replaceChildren();area.append(node("span","证据不足","badge"),node("h3","当前没有可批准的修复版本"),node("p",snapshot.release.reason));const ul=node("ul",undefined,"gate-list");for(const text of snapshot.release.missing)ul.append(node("li",text+"：待提供并核验"));area.append(ul,node("div","此页面为只读检查入口。暂不提供实验启动、权重替换或发布操作。","missing"));}
async function refresh(){const button=byId("refresh");button.disabled=true;byId("notice").textContent="正在读取项目证据…";try{const response=await fetch("/api/overview",{cache:"no-store"});if(!response.ok)throw new Error("证据服务暂不可用");snapshot=await response.json();byId("notice").textContent=snapshot.source_status==="VERIFIED_ARCHIVE"?"四个开发干预点已绑定归档证据。在线纠正尚未执行，成功率未测量；发布门槛保持关闭。":snapshot.source_error;renderCases();renderRuntime();renderRelease();byId("updated").textContent="页面读取："+new Date(snapshot.generated_utc).toLocaleString("zh-CN");}catch(error){snapshot=null;selected=null;byId("notice").textContent=error.message;byId("case-table").replaceChildren();detail(null);for(const id of ["runtime-detail","release-detail"]){byId(id).replaceChildren();byId(id).textContent="当前证据不可用，请重新读取。";}byId("updated").textContent="最近一次读取失败";}finally{button.disabled=false;}}
document.querySelectorAll(".tab").forEach(button=>button.addEventListener("click",()=>{document.querySelectorAll(".tab").forEach(b=>{const active=b===button;b.classList.toggle("active",active);b.setAttribute("aria-pressed",String(active));});document.querySelectorAll(".view").forEach(v=>v.hidden=v.id!==button.dataset.view);}));
byId("refresh").addEventListener("click",refresh);byId("task-filter").addEventListener("change",()=>snapshot&&renderCases());refresh();
