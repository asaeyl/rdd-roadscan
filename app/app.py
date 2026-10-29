import io, json, sqlite3, datetime, uuid
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st

ROOT=Path(__file__).resolve().parent                              # app/  (database + report photos stay here)
PROJECT_ROOT=ROOT.parent
FINAL_DIR=PROJECT_ROOT/"fine-tuning"/"final_model"                # the FINE-TUNED model: Crack, Pothole, Rut
MODEL=FINAL_DIR/"best.pt"
ARTIFACT=FINAL_DIR/"artifact.json"                                # image size + one confidence threshold per damage type
DB=ROOT/"roadscan.db"
PHOTO_DIR=ROOT/"report_images"
PHOTO_DIR.mkdir(exist_ok=True)

try:
    ARTIFACT_DATA=json.loads(ARTIFACT.read_text()) if ARTIFACT.exists() else {}
except Exception:
    ARTIFACT_DATA={}
IMGSZ=int(ARTIFACT_DATA.get("image_size",640))
THRESHOLDS={str(k):float(v) for k,v in (ARTIFACT_DATA.get("confidence_thresholds") or {}).items()}   # e.g. {"Crack":0.47,"Pothole":0.25,"Rut":0.41}
DEFAULT_CONF=0.25                                                  # only used if artifact.json has no thresholds

CLASSES={"Crack":("شق","Crack"),"Pothole":("حفرة","Pothole"),"Rut":("أخدود / تخدد","Rut")}      # the classes of the fine-tuned model
LEGACY={"D00":("شق طولي","Longitudinal Crack"),"D10":("شق عرضي","Transverse Crack"),
        "D20":("تشققات تمساحية","Alligator Crack"),"D40":("حفرة","Pothole")}                     # reports saved earlier with the old model
LABELS={**LEGACY,**CLASSES}
COLORS={"Crack":"#ff9100","Pothole":"#ff1744","Rut":"#00b8d4"}
TEXT_COLORS={"Crack":"#07120a","Pothole":"#ffffff","Rut":"#07120a"}
STATES=["جديد","قيد المراجعة","بانتظار الفحص","قيد الفحص","بانتظار قرار الصيانة","قيد الصيانة","مكتمل","مرفوض"]
st.set_page_config(page_title="RoadScan | Smart Road Maintenance",page_icon="🛣️",layout="wide",initial_sidebar_state="expanded")
st.markdown("""
<style>
:root{color-scheme:dark}
.stApp,[data-testid="stHeader"]{background:#080d10;color:#f4f8f5}
[data-testid="stSidebar"]{background:#111b1c;border-right:1px solid #263c33}
[data-testid="stSidebar"] *{color:#f5f9f5}
h1,h2,h3,p,label{color:#f1f8f2!important}
[data-testid="stMetric"]{background:#132120;padding:15px;border-radius:14px;border:1px solid #294637}
.stButton>button[kind="primary"],button[kind="primary"]{background:#86ed50!important;color:#07120a!important;border:0;font-weight:800}
.stButton>button{border-radius:12px}
[data-testid="stFileUploader"]{border:1px dashed #6fb947;border-radius:12px;padding:8px}
.hero{background:linear-gradient(115deg,#142c23,#10191a);border:1px solid #36533e;border-radius:20px;padding:26px;margin:8px 0 24px}
.hero h1{font-size:44px;margin:0;color:#fff!important}
.hero span{color:#8af052}
.hero p{color:#d5e3d8!important}
.tag{background:#274c2d;border-radius:25px;padding:7px 12px;color:#b7fb94;font-size:13px}
.panel{background:#142020;border:1px solid #2b4438;border-radius:14px;padding:16px;margin:10px 0}
.small{color:#c1cec5}
</style>""",unsafe_allow_html=True)

def ar_names(codes):
    """'Crack, Pothole' (or old 'D00, D40') -> Arabic names for display."""
    parts=[c.strip() for c in str(codes).split(",") if c.strip()]
    return "، ".join(LABELS.get(c,(c,c))[0] for c in parts) if parts else "غير محدد"

def dbconn():
    con=sqlite3.connect(DB,timeout=20)
    con.row_factory=sqlite3.Row
    return con
def initialize():
    with dbconn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS reports (
            id TEXT PRIMARY KEY,created TEXT,city TEXT,street TEXT,description TEXT,
            reporter TEXT,damage TEXT,confidence REAL,det_count INTEGER,
            status TEXT,priority TEXT,assigned_to TEXT,supervisor_note TEXT,
            inspection_note TEXT,inspection_result TEXT,image_file TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS events(
            id INTEGER PRIMARY KEY AUTOINCREMENT,report_id TEXT,time TEXT,actor TEXT,action TEXT,note TEXT)""")
def event(c,rid,actor,action,note=""):
    c.execute("INSERT INTO events(report_id,time,actor,action,note) VALUES(?,?,?,?,?)",
              (rid,datetime.datetime.now().isoformat(timespec="seconds"),actor,action,note))
def all_reports():
    with dbconn() as c:
        return pd.read_sql_query("SELECT * FROM reports ORDER BY created DESC",c)
def details(rid):
    with dbconn() as c:
        r=c.execute("SELECT * FROM reports WHERE id=?",(rid,)).fetchone()
        return dict(r) if r else None
def update(rid,actor,action,fields,note=""):
    allowed={"status","priority","assigned_to","supervisor_note","inspection_note","inspection_result"}
    if any(k not in allowed for k in fields): raise ValueError("Invalid update field")
    with dbconn() as c:
        if fields:
            cols=", ".join(f"{k}=?" for k in fields)
            c.execute(f"UPDATE reports SET {cols} WHERE id=?",(*fields.values(),rid))
        event(c,rid,actor,action,note)
def history(rid):
    with dbconn() as c:
        return pd.read_sql_query("SELECT time,actor,action,note FROM events WHERE report_id=? ORDER BY id DESC",c,params=(rid,))

if not MODEL.exists():
    st.error(f"Missing model file: {MODEL}. Fine-tune the model first and commit fine-tuning/final_model/best.pt and artifact.json.")
    st.stop()

@st.cache_resource(show_spinner="جاري تحميل النموذج بعد الضبط...")
def model():
    from ultralytics import YOLO
    m=YOLO(str(MODEL))
    names={str(v).strip().lower() for v in m.names.values()}
    if names!={k.lower() for k in CLASSES}:                         # e.g. the old D00-D40 weights were loaded by mistake
        raise RuntimeError(f"The model has the classes {sorted(m.names.values())}, but this app expects {list(CLASSES)}.")
    return m
def threshold_for(name):
    return THRESHOLDS.get(name,DEFAULT_CONF)
def analyze(im):
    m=model()
    floor=min(THRESHOLDS.values()) if THRESHOLDS else DEFAULT_CONF    # lowest confidence the model reports; each class is filtered below
    # PIL image on purpose: Ultralytics reads PIL as RGB, but a numpy array as BGR (the old call swapped the colour channels)
    result=m.predict(source=im.convert("RGB"),imgsz=IMGSZ,conf=floor,verbose=False)[0]
    detections=[]
    for b in result.boxes:
        idx=int(b.cls.item())
        raw=str(m.names[idx]).strip()
        name=next((k for k in CLASSES if k.lower()==raw.lower()),raw)
        score=float(b.conf.item())
        if score<threshold_for(name): continue                       # the threshold of THIS damage type (from artifact.json)
        detections.append({"name":name,"confidence":score,"box":list(map(int,b.xyxy[0].tolist()))})
    annotated=im.convert("RGB").copy()
    draw=ImageDraw.Draw(annotated)
    width=max(3,min(im.size)//150)
    for d in detections:
        x1,y1,x2,y2=d["box"]
        colour=COLORS.get(d["name"],"#8af052")
        draw.rectangle([x1,y1,x2,y2],outline=colour,width=width)
        txt=f'{d["name"]} {d["confidence"]:.0%}'
        draw.rectangle([x1,max(0,y1-24),x1+max(110,len(txt)*10),y1],fill=colour)
        draw.text((x1+4,max(0,y1-22)),txt,fill=TEXT_COLORS.get(d["name"],"#07120a"))
    return annotated,detections

initialize()
with st.sidebar:
    st.markdown("## 🛣️ RoadScan")
    st.caption("SMART ROAD MAINTENANCE")
    role=st.radio("الدخول حسب الدور",["المواطن","مشرف الصيانة","فريق الفحص"])
    if role=="المواطن": page=st.radio("القائمة",["بلاغ جديد","متابعة بلاغ"])
    elif role=="مشرف الصيانة": page=st.radio("القائمة",["لوحة الصيانة","مراجعة البلاغات"])
    else: page=st.radio("القائمة",["مهام الفحص","تحليل صور متعددة"])
    st.divider()
    st.caption("نسخة معاينة: اختيار الأدوار متاح للجميع، دون تسجيل دخول أو صلاحيات فعلية.")
    st.caption(f"النموذج: بعد الضبط (Crack, Pothole, Rut) • {IMGSZ}px")
    if THRESHOLDS: st.caption("حدود الثقة: "+" • ".join(f"{k} {v:.2f}" for k,v in THRESHOLDS.items()))
st.markdown("""<div class="hero"><h1>Road<span>Scan</span></h1>
<p>نظام ذكي لرصد أضرار الطرق وإدارة البلاغات والفحص والصيانة</p>
<span class="tag">● AI-Powered Road Monitoring</span></div>""",unsafe_allow_html=True)

if role=="المواطن" and page=="بلاغ جديد":
    st.header("📸 إنشاء بلاغ عن ضرر في الطريق")
    file=st.file_uploader("صورة الطريق",type=["jpg","jpeg","png","webp"])
    if file:
        try:
            im=Image.open(file).convert("RGB")
            with st.spinner("جاري تحليل الصورة..."):
                annotated,ds=analyze(im)
            a,b=st.columns([1.8,1])
            with a:
                st.image(annotated,caption="نتيجة الكشف",use_container_width=True)
                with st.expander("الصورة الأصلية"): st.image(im,use_container_width=True)
            with b:
                st.metric("عدد الأضرار",len(ds))
                if not ds: st.info("لم تُكتشف أضرار فوق حد الثقة. يمكن إرسال البلاغ للمراجعة.")
                for d in ds:
                    label=LABELS.get(d["name"],("ضرر غير معروف","Unknown"))[0]
                    st.markdown(f'<div class="panel"><b style="color:{COLORS.get(d["name"],"#8af052")}">{d["name"]}</b> · {label}<br>الثقة: {d["confidence"]:.0%}</div>',unsafe_allow_html=True)
            st.session_state["current_detection"]=(file.name,annotated,ds)
        except Exception as e:
            st.error(f"فشل التحليل: {e}")
            st.session_state.pop("current_detection",None)
    with st.form("new_report"):
        st.subheader("تفاصيل البلاغ")
        city=st.text_input("المدينة *")
        street=st.text_input("الحي / اسم الشارع / وصف الموقع *")
        reporter=st.text_input("اسم المبلّغ (اختياري)")
        description=st.text_area("وصف إضافي")
        send=st.form_submit_button("إرسال البلاغ",type="primary")
    if send:
        current=st.session_state.get("current_detection")
        if not file or not current or current[0]!=file.name:
            st.error("ارفعي صورة وانتظري انتهاء التحليل أولًا.")
        elif not city.strip() or not street.strip():
            st.warning("المدينة والموقع مطلوبان.")
        else:
            _,annotated,ds=current
            rid="RS-"+datetime.datetime.now().strftime("%y%m%d")+"-"+uuid.uuid4().hex[:6].upper()
            img_name=rid+".jpg"
            annotated.save(PHOTO_DIR/img_name,quality=88)
            codes=[d["name"] for d in ds]
            priority="تحتاج تقييم"  # only supervisor can set priority
            with dbconn() as c:
                c.execute("""INSERT INTO reports
                (id,created,city,street,description,reporter,damage,confidence,det_count,
                status,priority,assigned_to,supervisor_note,inspection_note,inspection_result,image_file)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (rid,datetime.datetime.now().isoformat(timespec="seconds"),city,street,description,reporter,
                 ", ".join(codes) if codes else "غير محدد",
                 max((d["confidence"] for d in ds),default=0),len(ds),
                 "جديد",priority,"","","","",img_name))
                event(c,rid,"المواطن","إنشاء بلاغ")
            st.success(f"تم تسجيل البلاغ. رقم المتابعة: {rid}")
            st.code(rid)
elif role=="المواطن" and page=="متابعة بلاغ":
    st.header("🔍 متابعة حالة البلاغ")
    rid=st.text_input("أدخلي رقم البلاغ",placeholder="RS-...")
    if rid:
        r=details(rid.strip().upper())
        if r:
            st.metric("حالة البلاغ",r["status"])
            st.write("رقم البلاغ:",r["id"])
            st.write("الموقع:",r["city"],"—",r["street"])
            st.write("الأضرار:",ar_names(r["damage"]))
            st.dataframe(history(r["id"])[["time","action"]],hide_index=True,use_container_width=True)
        else: st.warning("لم يتم العثور على هذا الرقم.")
elif role=="مشرف الصيانة":
    df=all_reports()
    if page=="لوحة الصيانة":
        st.header("📊 لوحة متابعة الصيانة")
        a,b,c,d=st.columns(4)
        a.metric("إجمالي البلاغات",len(df))
        b.metric("الجديدة",int((df.status=="جديد").sum()) if len(df) else 0)
        c.metric("بانتظار الفحص",int((df.status=="بانتظار الفحص").sum()) if len(df) else 0)
        d.metric("المكتملة",int((df.status=="مكتمل").sum()) if len(df) else 0)
        if not df.empty:
            view=df[["id","city","street","damage","priority","status","assigned_to"]].copy()
            view["damage"]=view["damage"].map(ar_names)
            st.dataframe(view,hide_index=True,use_container_width=True)
            st.download_button("تنزيل سجل البلاغات CSV",df.to_csv(index=False).encode("utf-8-sig"),file_name="roadscan_reports.csv",mime="text/csv")
    else:
        st.header("🗂️ مراجعة البلاغات وإسناد المهام")
        if df.empty: st.info("لا توجد بلاغات حتى الآن.")
        else:
            rid=st.selectbox("اختاري بلاغًا",df.id.tolist())
            r=details(rid)
            st.write("الموقع:",r["city"],"—",r["street"])
            st.write("الضرر المكتشف:",ar_names(r["damage"]))
            st.write("الحالة الحالية:",r["status"])
            p=PHOTO_DIR/r["image_file"]
            if p.exists(): st.image(str(p),width=550)
            with st.form("supervisor_form"):
                priority=st.selectbox("الأولوية بعد المراجعة",["تحتاج تقييم","منخفضة","متوسطة","عالية","طارئة"],
                                      index=["تحتاج تقييم","منخفضة","متوسطة","عالية","طارئة"].index(r["priority"]) if r["priority"] in ["تحتاج تقييم","منخفضة","متوسطة","عالية","طارئة"] else 0)
                assigned=st.text_input("اسم فريق الفحص / رمز الفريق",value=r["assigned_to"])
                status=st.selectbox("حالة البلاغ",STATES,index=STATES.index(r["status"]) if r["status"] in STATES else 0)
                note=st.text_area("ملاحظات المشرف",value=r["supervisor_note"])
                saved=st.form_submit_button("حفظ قرار المشرف",type="primary")
            if saved:
                update(rid,"مشرف الصيانة","مراجعة وإسناد",
                       {"priority":priority,"assigned_to":assigned,"status":status,"supervisor_note":note},note)
                st.success("تم تحديث البلاغ")
                st.rerun()
            with st.expander("سجل الإجراءات"): st.dataframe(history(rid),hide_index=True)
elif role=="فريق الفحص" and page=="مهام الفحص":
    st.header("🦺 مهام الفحص الميداني")
    df=all_reports()
    tasks=df[df.status.isin(["بانتظار الفحص","قيد الفحص","بانتظار قرار الصيانة"])] if not df.empty else df
    if tasks.empty: st.info("لا توجد مهام فحص مسندة حاليًا. على المشرف إسناد بلاغ وتغيير حالته إلى «بانتظار الفحص».")
    else:
        st.dataframe(tasks[["id","city","street","assigned_to","priority","status"]],hide_index=True,use_container_width=True)
        rid=st.selectbox("اختاري المهمة",tasks.id.tolist())
        r=details(rid)
        st.write("الموقع:",r["city"],"—",r["street"])
        p=PHOTO_DIR/r["image_file"]
        if p.exists(): st.image(str(p),width=550)
        with st.form("inspection_form"):
            result=st.selectbox("نتيجة المعاينة",["تم تأكيد الضرر","يحتاج إعادة تقييم","لا يوجد ضرر","تعذر الوصول للموقع"])
            notes=st.text_area("ملاحظات المعاينة وتوصيات الإصلاح")
            status=st.selectbox("حالة المهمة",["قيد الفحص","بانتظار قرار الصيانة"])
            submit=st.form_submit_button("إرسال تقرير الفحص",type="primary")
        if submit:
            update(rid,"فريق الفحص","تقرير فحص",
                   {"inspection_result":result,"inspection_note":notes,"status":status},notes)
            st.success("تم حفظ تقرير الفحص")
            st.rerun()
else:
    st.header("📷 تحليل صور متعددة")
    files=st.file_uploader("ارفعي صور الفحص",type=["jpg","jpeg","png","webp"],accept_multiple_files=True)
    rows=[]
    if files:
        for file in files:
            try:
                im=Image.open(file).convert("RGB")
                annotated,ds=analyze(im)
                st.markdown(f"#### {file.name}")
                st.image(annotated,width=650)
                for d in ds:
                    rows.append({"image":file.name,"damage":d["name"],"confidence":round(d["confidence"],3)})
            except Exception as e: st.error(f"{file.name}: {e}")
        if rows:
            st.download_button("تنزيل CSV",pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"),file_name="inspection_detections.csv",mime="text/csv")
        else: st.info("لم تُكتشف أضرار في الصور المرفوعة عند حدود الثقة الحالية.")
st.divider()
st.caption("RoadScan • نسخة اختبار إجراءات العمل. لا توجد مصادقة أو تخزين سحابي دائم؛ لا تدخلي بيانات شخصية حقيقية.")
