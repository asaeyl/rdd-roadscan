from pathlib import Path
from io import BytesIO
from datetime import datetime
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title='RoadScan | Smart Road Intelligence', page_icon='🛣️', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Tajawal:wght@400;500;700;800&display=swap');
html,body,[class*="css"],.stApp {font-family:'Inter','Tajawal',sans-serif}
.stApp {background:linear-gradient(135deg,#0b1420 0%,#142436 60%,#0c1a29 100%);color:#ecf4f7}
[data-testid="stSidebar"] {background:#0d1b2a;border-right:1px solid #294354}
[data-testid="stSidebar"] * {color:#e7f2f4}
h1,h2,h3 {color:#f5fafc!important;letter-spacing:-.025em}
p,span,label {color:inherit}
.hero {background:linear-gradient(115deg,#183246,#10202f);border:1px solid #315267;border-radius:22px;padding:28px 30px;margin-bottom:22px}
.eyebrow {color:#47d8c4;letter-spacing:.19em;font-weight:800;font-size:12px}
.hero h1 {font-size:clamp(32px,4vw,50px);margin:10px 0 8px!important}
.hero p {color:#bbccd5;font-size:15px;margin:0}
.chip {display:inline-block;border:1px solid #39786f;background:#173e40;color:#74ead8;border-radius:30px;padding:5px 12px;font-size:12px;font-weight:700;margin-top:14px}
.metric {border:1px solid #315066;border-radius:16px;background:#14283a;padding:18px;min-height:104px}
.metric .number {font-size:29px;font-weight:800;color:#f3fafb}
.metric .label {font-size:13px;color:#a7c1cc}
.stButton>button[kind="primary"],.stDownloadButton>button[kind="primary"] {background:#22b8a7;color:#071d25;border:0;border-radius:11px;font-weight:800}
.stButton>button,.stDownloadButton>button {border-radius:11px}
[data-testid="stFileUploader"] {border:1px dashed #4c8d92;border-radius:14px;padding:8px}
[data-testid="stAlert"] {border-radius:12px}
hr {border-color:#314a5b}
</style>''',unsafe_allow_html=True)

T = {
 'English': {'subtitle':'AI-powered road damage detection & inspection','choose':'Workspace','citizen':'Citizen','supervisor':'Maintenance supervisor','inspection':'Inspection team','upload':'Upload road image(s)','original':'Original image','result':'AI detection','none':'No damage detected at this threshold. This does not confirm that the road is damage-free.','det':'Detections','images':'Images analyzed','types':'Damage types','confidence':'Confidence threshold','download':'Download annotated image','csv':'Download inspection report (CSV)','notice':'Priority levels are experimental suggestions based on detected damage type and box size, not verified road safety assessments.','report':'Local demo reports (this session only)','submit':'Save demo report','saved':'Demo report saved in this browser session. No server database is connected.','about':'RoadScan demo • YOLO final weights • 640px inference','welcome':'Upload a road photo to start the analysis.','severity':'Suggested priority','high':'High','medium':'Medium','low':'Low','detail':'Detection details','lang':'Language'},
 'العربية': {'subtitle':'نظام ذكي لاكتشاف أضرار الطرق وفحصها','choose':'واجهة المستخدم','citizen':'المواطن','supervisor':'مشرف الصيانة','inspection':'فريق الفحص','upload':'ارفعي صورة أو عدة صور للطريق','original':'الصورة الأصلية','result':'نتيجة الكشف','none':'لم يُكتشف ضرر عند حد الثقة الحالي، وهذا لا يعني أن الطريق خالٍ من الأضرار.','det':'الاكتشافات','images':'الصور المحللة','types':'أنواع الأضرار','confidence':'حد الثقة','download':'تحميل الصورة مع المربعات','csv':'تحميل تقرير الفحص CSV','notice':'الأولويات المقترحة تجريبية بناءً على نوع الضرر وحجم المربع، وليست تقييمًا معتمدًا لسلامة الطريق.','report':'البلاغات التجريبية (خلال الجلسة فقط)','submit':'حفظ بلاغ تجريبي','saved':'تم حفظ البلاغ داخل جلسة التطبيق فقط، ولا توجد قاعدة بيانات دائمة.','about':'نسخة تجريبية • النموذج النهائي YOLO • حجم الصورة 640','welcome':'ارفعي صورة شارع لبدء التحليل.','severity':'الأولوية المقترحة','high':'عالية','medium':'متوسطة','low':'منخفضة','detail':'تفاصيل الاكتشاف','lang':'اللغة'}
}
LABELS={'D00':('Longitudinal crack','شق طولي'),'D10':('Transverse crack','شق عرضي'),'D20':('Alligator crack','تشققات تمساحية'),'D40':('Pothole','حفرة')}

with st.sidebar:
    st.markdown('## 🛣️ RoadScan')
    language=st.selectbox('Language / اللغة',['English','العربية'])
    t=T[language]
    role=st.radio(t['choose'],[t['citizen'],t['supervisor'],t['inspection']])
    conf=st.slider(t['confidence'],0.10,0.90,0.25,0.05)
    st.caption(t['about'])

st.markdown(f'''<div class="hero"><div class="eyebrow">ROADSCAN / INTELLIGENT INFRASTRUCTURE</div><h1>RoadScan <span style="color:#47d8c4">AI</span></h1><p>{t['subtitle']}</p><span class="chip">● FINAL MODEL ACTIVE</span></div>''',unsafe_allow_html=True)
weights=Path(__file__).resolve().parent/'models'/'best.pt'
if not weights.exists():
    st.error(f'Missing model file: {weights}')
    st.stop()
@st.cache_resource(show_spinner='Loading road detection model...')
def get_model(path):
    return YOLO(str(path))
try:
    model=get_model(str(weights))
except Exception as e:
    st.error(f'Model could not be loaded: {e}')
    st.stop()

multi=(role==t['inspection'])
uploads=st.file_uploader(t['upload'],type=['jpg','jpeg','png','webp'],accept_multiple_files=multi)
if 'reports' not in st.session_state: st.session_state.reports=[]
files=uploads if isinstance(uploads,list) else ([uploads] if uploads else [])
summary=[]
if files:
    all_detections=[]
    for uploaded in files:
        try:
            img=Image.open(uploaded).convert('RGB')
            res=model.predict(img,imgsz=640,conf=conf,verbose=False)[0]
            annotated=Image.fromarray(res.plot()[...,::-1])
        except Exception as e:
            st.error(f'{uploaded.name}: {e}')
            continue
        records=[]
        for box in res.boxes:
            cls=int(box.cls[0].item())
            name=str(res.names[cls])
            score=float(box.conf[0].item())
            xy=box.xyxy[0].tolist()
            area=max(0,xy[2]-xy[0])*max(0,xy[3]-xy[1])/(img.width*img.height)
            # Display-only heuristic; not a validated engineering risk assessment.
            priority='High' if name=='D40' or (name=='D20' and area>0.06) else ('Medium' if name in ('D20','D00') else 'Low')
            records.append({'image':uploaded.name,'damage_code':name,'damage_name':LABELS.get(name,(name,name))[1 if language=='العربية' else 0], 'confidence':round(score,3),'box_area_fraction':round(area,4),'suggested_priority':priority})
        summary.append({'image':uploaded.name,'detections':len(records),'damage_types':', '.join(sorted(set(r['damage_code'] for r in records))) or 'None'})
        all_detections.extend(records)
        st.markdown(f'### {uploaded.name}')
        c1,c2=st.columns(2)
        with c1: st.image(img,caption=t['original'],use_container_width=True)
        with c2: st.image(annotated,caption=t['result'],use_container_width=True)
        if records:
            st.dataframe(pd.DataFrame(records),use_container_width=True,hide_index=True)
        else: st.info(t['none'])
        buf=BytesIO();annotated.save(buf,format='PNG')
        st.download_button('⬇️ '+t['download'],data=buf.getvalue(),file_name=Path(uploaded.name).stem+'_roadscan.png',mime='image/png',key='dl_'+uploaded.name)
        if role==t['citizen']:
            if st.button('📌 '+t['submit'],key='save_'+uploaded.name):
                st.session_state.reports.append({'image':uploaded.name,'detections':len(records),'timestamp':datetime.now().isoformat(timespec='seconds'),'status':'Demo only'})
                st.success(t['saved'])
        st.divider()
    a,b,c=st.columns(3)
    with a: st.markdown(f'<div class="metric"><div class="number">{len(summary)}</div><div class="label">{t["images"]}</div></div>',unsafe_allow_html=True)
    with b: st.markdown(f'<div class="metric"><div class="number">{len(all_detections)}</div><div class="label">{t["det"]}</div></div>',unsafe_allow_html=True)
    with c: st.markdown(f'<div class="metric"><div class="number">{len(set(r["damage_code"] for r in all_detections))}</div><div class="label">{t["types"]}</div></div>',unsafe_allow_html=True)
    if role in (t['supervisor'],t['inspection']):
        st.warning(t['notice'])
        if all_detections:
            priority_order={'High':0,'Medium':1,'Low':2}
            ranked=sorted(all_detections,key=lambda r:(priority_order[r['suggested_priority']],-r['confidence']))
            st.markdown('### '+t['severity'])
            st.dataframe(pd.DataFrame(ranked),hide_index=True,use_container_width=True)
        df=pd.DataFrame(all_detections if all_detections else summary)
        st.download_button('⬇️ '+t['csv'],df.to_csv(index=False).encode('utf-8-sig'),'roadscan_inspection.csv','text/csv')
else:
    st.info(t['welcome'])
if role==t['supervisor']:
    st.markdown('### '+t['report'])
    if st.session_state.reports: st.dataframe(pd.DataFrame(st.session_state.reports),hide_index=True,use_container_width=True)
    else: st.caption('No demo reports in this session yet.')
st.caption('RoadScan • Research prototype. Detection output must be reviewed by a human before maintenance decisions.')
