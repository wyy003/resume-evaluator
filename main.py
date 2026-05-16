from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from datetime import datetime
import sqlite3
import json
from pypdf import PdfReader
import re
import random

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 初始化数据库
def init_db():
    conn = sqlite3.connect('resumes.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS uploads
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  filename TEXT,
                  upload_time TEXT,
                  parsed_data TEXT,
                  jd_text TEXT,
                  llm_analysis TEXT)''')
    conn.commit()
    conn.close()

init_db()

# LLM 分析函数（模拟版本 + 随机波动 + 优化评分逻辑）
def analyze_with_llm_mock(resume_json, jd_text):
    """模拟 LLM 分析，返回固定格式的结果（带随机波动 + 智能评分）"""

    sections = resume_json.get("sections", {})
    base_score = 50  # 降低基础分

    # 1. 段落完整性（最多 20 分）
    if sections.get("education") and sections["education"].strip():
        base_score += 5
    if sections.get("work") and sections["work"].strip():
        base_score += 8
    if sections.get("skills") and sections["skills"].strip():
        base_score += 4
    if sections.get("projects") and sections["projects"].strip():
        base_score += 3

    # 2. 关键词匹配（最多 30 分）
    keyword_score = 0
    if jd_text.strip():
        jd_keywords = extract_keywords(jd_text)
        match_count = count_keyword_matches(resume_json, jd_keywords)
        keyword_score = min(match_count * 3, 30)
        base_score += keyword_score

    # 3. 内容质量（最多 20 分）
    quality_score = 0

    # 工作经历有数字（量化）
    if has_numbers(sections.get("work", "")):
        quality_score += 8

    # 项目描述详细（>200 字）
    if len(sections.get("projects", "")) > 200:
        quality_score += 7

    # 技能数量（>5 个）
    if count_skills(sections.get("skills", "")) > 5:
        quality_score += 5

    base_score += quality_score

    # 4. 随机波动 ±5%
    fluctuation = random.randint(-5, 5)
    final_score = min(max(base_score + fluctuation, 0), 100)

    # 5. 动态生成优势/不足/建议
    strengths, weaknesses, suggestions = generate_feedback(
        sections, jd_text, keyword_score, quality_score
    )

    return {
        "match_score": final_score,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "suggestions": suggestions
    }

# 辅助函数
def extract_keywords(jd_text):
    """从 JD 中提取关键词"""
    keywords = []
    common_tech = ["python", "java", "javascript", "react", "vue", "django",
                   "fastapi", "flask", "sql", "mysql", "postgresql", "mongodb",
                   "docker", "kubernetes", "aws", "git", "linux", "redis",
                   "typescript", "node", "express", "spring", "golang", "rust",
                   "tensorflow", "pytorch", "机器学习", "深度学习", "ai", "nlp"]

    jd_lower = jd_text.lower()
    for tech in common_tech:
        if tech in jd_lower:
            keywords.append(tech)

    return keywords

def count_keyword_matches(resume_json, keywords):
    """计算简历中匹配的关键词数量"""
    resume_text = json.dumps(resume_json, ensure_ascii=False).lower()
    match_count = 0
    for keyword in keywords:
        if keyword in resume_text:
            match_count += 1
    return match_count

def has_numbers(text):
    """检查文本中是否有数字（量化指标）"""
    return bool(re.search(r'\d+', text))

def count_skills(skills_text):
    """统计技能数量（简单按分隔符分割）"""
    if not skills_text:
        return 0
    separators = [',', '、', '，', ';', '；', '\n']
    count = 1
    for sep in separators:
        count = max(count, skills_text.count(sep) + 1)
    return count

def generate_feedback(sections, jd_text, keyword_score, quality_score):
    """动态生成优势/不足/建议"""
    strengths = []
    weaknesses = []
    suggestions = []

    # 教育背景
    if sections.get("education") and sections["education"].strip():
        strengths.append("教育背景清晰完整")
    else:
        weaknesses.append("缺少教育背景信息")
        suggestions.append("建议补充教育经历")

    # 工作经历
    if sections.get("work") and sections["work"].strip():
        if has_numbers(sections["work"]):
            strengths.append("工作经历包含量化数据，展示具体成果")
        else:
            weaknesses.append("工作成果缺少量化数据")
            suggestions.append("建议用数字量化工作成果（如：提升 30% 效率、负责 5 人团队）")
    else:
        weaknesses.append("缺少工作经历")
        suggestions.append("建议补充相关工作经验")

    # 关键词匹配
    if jd_text.strip():
        if keyword_score > 15:
            strengths.append("技能与岗位要求匹配度较高")
        elif keyword_score > 5:
            weaknesses.append("部分技能与岗位要求匹配，但覆盖不全")
            suggestions.append("建议补充更多与 JD 相关的技能关键词")
        else:
            weaknesses.append("简历中缺少岗位相关的关键技能")
            suggestions.append("建议仔细阅读 JD，在简历中突出相关技能和经验")

    # 项目经历
    if sections.get("projects") and len(sections["projects"]) > 200:
        strengths.append("项目经验描述详细，体现实践能力")
    elif sections.get("projects") and sections["projects"].strip():
        weaknesses.append("项目描述较为简略")
        suggestions.append("建议采用 STAR 法则详细描述项目（情境-任务-行动-结果）")
    else:
        weaknesses.append("缺少项目经历")
        suggestions.append("建议补充相关项目经验")

    # 技能部分
    skill_count = count_skills(sections.get("skills", ""))
    if skill_count > 5:
        strengths.append(f"技能列表丰富（{skill_count} 项技能）")
    elif skill_count > 0:
        suggestions.append("可以补充更多专业技能，展示技术广度")

    # 确保至少有内容
    if not strengths:
        strengths.append("简历基本信息完整")
    if not weaknesses:
        weaknesses.append("整体表现良好，可进一步优化细节")
    if not suggestions:
        suggestions.append("保持简历更新，持续积累项目经验")

    return strengths, weaknesses, suggestions

# PDF 解析函数
def parse_pdf(file_path):
    """提取 PDF 文字并识别简历段落"""
    try:
        reader = PdfReader(file_path)
        full_text = ""

        # 提取所有页面的文字
        for page in reader.pages:
            full_text += page.extract_text() + "\n"

        if not full_text.strip():
            return {
                "filename": os.path.basename(file_path),
                "text": "",
                "sections": {
                    "education": "",
                    "work": "",
                    "skills": "",
                    "projects": ""
                },
                "error": "无法提取文字，可能是扫描版 PDF"
            }

        lines = full_text.split('\n')

        section_keywords = {
            "education": ["教育", "学历", "education", "academic"],
            "work": ["工作", "经历", "experience", "employment", "职位"],
            "skills": ["技能", "skills", "能力", "专长"],
            "projects": ["项目", "project", "作品"]
        }

        # 识别标题位置
        title_line_indices = {
            "education": None,
            "work": None,
            "skills": None,
            "projects": None
        }

        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if not line_lower:
                continue

            for section, keywords in section_keywords.items():
                if any(keyword in line_lower for keyword in keywords):
                    if title_line_indices[section] is None:
                        title_line_indices[section] = i

        # 提取段落内容
        sections = {
            "education": "",
            "work": "",
            "skills": "",
            "projects": ""
        }

        # 获取所有标题的行号，按行号排序
        all_title_indices = []
        for section, idx in title_line_indices.items():
            if idx is not None:
                all_title_indices.append((idx, section))
        all_title_indices.sort()

        # 提取每个段落的内容
        for i, (start_idx, section) in enumerate(all_title_indices):
            # 确定结束位置：下一个标题的位置，或文件末尾
            if i + 1 < len(all_title_indices):
                end_idx = all_title_indices[i + 1][0]
            else:
                end_idx = len(lines)

            # 提取内容（跳过标题行本身）
            content_lines = lines[start_idx + 1:end_idx]
            sections[section] = "\n".join(content_lines).strip()

        return {
            "filename": os.path.basename(file_path),
            "text": full_text,
            "sections": sections
        }

    except Exception as e:
        return {
            "filename": os.path.basename(file_path),
            "text": "",
            "sections": {
                "education": "",
                "work": "",
                "skills": "",
                "projects": ""
            },
            "error": str(e)
        }

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # 获取历史记录
    conn = sqlite3.connect('resumes.db')
    c = conn.cursor()
    c.execute('SELECT id, filename, upload_time, parsed_data FROM uploads ORDER BY id DESC')
    history = c.fetchall()
    conn.close()

    return templates.TemplateResponse("index.html", {
        "request": request,
        "history": history
    })

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), jd_text: str = Form("")):
    # 保存文件
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # 解析 PDF
    parsed_data = parse_pdf(file_path)
    parsed_json = json.dumps(parsed_data, ensure_ascii=False)

    # LLM 分析
    llm_analysis = None
    if jd_text.strip():
        llm_analysis = analyze_with_llm_mock(parsed_data, jd_text)
        llm_analysis_json = json.dumps(llm_analysis, ensure_ascii=False)
    else:
        llm_analysis_json = None

    # 保存到数据库
    conn = sqlite3.connect('resumes.db')
    c = conn.cursor()
    upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('INSERT INTO uploads (filename, upload_time, parsed_data, jd_text, llm_analysis) VALUES (?, ?, ?, ?, ?)',
              (file.filename, upload_time, parsed_json, jd_text, llm_analysis_json))
    upload_id = c.lastrowid
    conn.commit()
    conn.close()

    print(f"✅ 收到文件: {file.filename}")
    print(f"📄 解析结果: {len(parsed_data.get('text', ''))} 字符")
    if llm_analysis:
        print(f"🤖 LLM 分析: 匹配度 {llm_analysis['match_score']}%")

    return {
        "message": "上传成功",
        "filename": file.filename,
        "upload_id": upload_id,
        "parsed_data": parsed_data,
        "llm_analysis": llm_analysis
    }

@app.get("/result/{upload_id}")
async def get_result(request: Request, upload_id: int):
    """查看解析结果详情页"""
    conn = sqlite3.connect('resumes.db')
    c = conn.cursor()
    c.execute('SELECT filename, upload_time, parsed_data, jd_text, llm_analysis FROM uploads WHERE id = ?', (upload_id,))
    result = c.fetchone()
    conn.close()

    if not result:
        return HTMLResponse("未找到记录", status_code=404)

    filename, upload_time, parsed_json, jd_text, llm_analysis_json = result
    parsed_data = json.loads(parsed_json) if parsed_json else {}
    llm_analysis = json.loads(llm_analysis_json) if llm_analysis_json else None

    return templates.TemplateResponse("result.html", {
        "request": request,
        "filename": filename,
        "upload_time": upload_time,
        "parsed_data": parsed_data,
        "jd_text": jd_text,
        "llm_analysis": llm_analysis
    })
