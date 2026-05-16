from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from datetime import datetime
import sqlite3
import json
from pypdf import PdfReader
from docx import Document
import re
import random
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

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
                  llm_analysis TEXT,
                  tokens_used INTEGER,
                  api_cost REAL)''')
    conn.commit()
    conn.close()

init_db()

# 初始化 OpenAI 客户端
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# LLM 分析函数（真实 OpenAI API）
def analyze_with_llm(resume_json, jd_text):
    """使用 OpenAI API 分析简历与 JD 的匹配度"""

    # 构建优化后的 Prompt
    prompt = f"""你是一位资深 HR 和招聘专家，请深度分析以下简历与职位描述的匹配度。

【职位描述】
{jd_text}

【简历内容】
教育背景：{resume_json.get('sections', {}).get('education', '无')}

工作经历：{resume_json.get('sections', {}).get('work', '无')}

技能特长：{resume_json.get('sections', {}).get('skills', '无')}

项目经验：{resume_json.get('sections', {}).get('projects', '无')}

【评分标准】（总分 100 分）
1. 教育背景匹配度（20分）：学历层次、专业相关性、院校背景
2. 工作经验匹配度（35分）：年限要求、行业经验、岗位相关性、职责匹配
3. 技能匹配度（25分）：必备技能覆盖率、技能熟练度、技术栈匹配
4. 项目经验匹配度（20分）：项目复杂度、业务场景相似度、成果量化

【分析要求】
1. match_score：基于上述标准给出 0-100 的综合评分
2. strengths：列出 3-5 个核心优势，每条需包含：
   - 具体的匹配点（引用简历原文）
   - 与 JD 的对应关系
   - 为什么这是优势
3. weaknesses：列出 2-4 个明显不足，每条需包含：
   - 具体缺失的要求（引用 JD 原文）
   - 当前简历的差距
   - 对求职的影响程度
4. suggestions：列出 3-5 个可操作的改进建议，每条需包含：
   - 具体的改进方向
   - 如何补充或优化
   - 预期的提升效果

【输出格式】
只返回以下 JSON 格式，不要任何其他文字：
{{
  "match_score": 75,
  "strengths": [
    "具有3年Python开发经验，与JD要求的'2年以上Python经验'高度匹配，且简历中提到使用FastAPI框架，正是岗位技术栈",
    "项目经验中的'电商推荐系统'与JD中的'推荐算法优化'业务场景一致，展示了相关领域的实战能力"
  ],
  "weaknesses": [
    "JD要求'熟悉Docker/K8s容器化部署'，但简历技能部分未提及相关经验，可能在DevOps能力上存在短板",
    "工作经历缺少量化成果（如性能提升百分比、用户增长数据），难以体现实际业务价值"
  ],
  "suggestions": [
    "在技能部分补充Docker和Kubernetes经验，如果有相关实践可详细描述部署流程和遇到的问题",
    "为每个项目添加量化指标，如'优化推荐算法使点击率提升15%'，增强说服力",
    "工作经历按STAR法则重写（情境-任务-行动-结果），突出解决问题的能力和业务影响"
  ]
}}"""

    try:
        # 调用 OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一位专业的 HR，擅长分析简历与职位的匹配度。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )

        # 解析返回结果
        result_text = response.choices[0].message.content.strip()

        # 尝试提取 JSON（可能包含 markdown 代码块）
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()

        result = json.loads(result_text)

        # 验证返回格式
        if not all(key in result for key in ["match_score", "strengths", "weaknesses", "suggestions"]):
            raise ValueError("返回格式不完整")

        # 添加 Token 使用信息
        result['tokens_used'] = response.usage.total_tokens
        result['prompt_tokens'] = response.usage.prompt_tokens
        result['completion_tokens'] = response.usage.completion_tokens
        # GPT-3.5-turbo 价格：$0.0015/1K prompt tokens, $0.002/1K completion tokens
        result['api_cost'] = (response.usage.prompt_tokens * 0.0015 + response.usage.completion_tokens * 0.002) / 1000

        return result

    except Exception as e:
        print(f"LLM 分析失败: {e}")
        # 降级到 Mock 函数
        return analyze_with_llm_mock(resume_json, jd_text)

# LLM 分析函数（Mock 版本，作为降级方案）
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

# DOCX 解析函数
def parse_docx(file_path):
    """提取 DOCX 文字并识别简历段落"""
    try:
        doc = Document(file_path)
        full_text = ""

        # 提取所有段落的文字
        for paragraph in doc.paragraphs:
            full_text += paragraph.text + "\n"

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
                "error": "无法提取文字，文档可能为空"
            }

        # 使用与 PDF 相同的段落识别逻辑
        return parse_text_sections(full_text, os.path.basename(file_path))

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

        return parse_text_sections(full_text, os.path.basename(file_path))

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

# 通用文本段落识别函数
def parse_text_sections(full_text, filename):
    """从文本中识别简历段落"""
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
        "filename": filename,
        "text": full_text,
        "sections": sections
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

    # 根据文件类型选择解析函数
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext == '.pdf':
        parsed_data = parse_pdf(file_path)
    elif file_ext in ['.docx', '.doc']:
        parsed_data = parse_docx(file_path)
    else:
        return JSONResponse(
            status_code=400,
            content={"error": f"不支持的文件格式: {file_ext}，仅支持 PDF 和 DOCX"}
        )

    parsed_json = json.dumps(parsed_data, ensure_ascii=False)

    # LLM 分析（使用真实 OpenAI API）
    llm_analysis = None
    if jd_text.strip():
        llm_analysis = analyze_with_llm(parsed_data, jd_text)
        llm_analysis_json = json.dumps(llm_analysis, ensure_ascii=False)
    else:
        llm_analysis_json = None

    # 保存到数据库
    conn = sqlite3.connect('resumes.db')
    c = conn.cursor()
    upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 提取 Token 和成本信息
    tokens_used = llm_analysis.get('tokens_used', 0) if llm_analysis else 0
    api_cost = llm_analysis.get('api_cost', 0.0) if llm_analysis else 0.0

    c.execute('INSERT INTO uploads (filename, upload_time, parsed_data, jd_text, llm_analysis, tokens_used, api_cost) VALUES (?, ?, ?, ?, ?, ?, ?)',
              (file.filename, upload_time, parsed_json, jd_text, llm_analysis_json, tokens_used, api_cost))
    upload_id = c.lastrowid
    conn.commit()
    conn.close()

    print(f"✅ 收到文件: {file.filename}")
    print(f"📄 解析结果: {len(parsed_data.get('text', ''))} 字符")
    if llm_analysis:
        print(f"🤖 LLM 分析: 匹配度 {llm_analysis['match_score']}%")
        if tokens_used > 0:
            print(f"💰 Token 使用: {tokens_used} tokens (${api_cost:.4f})")

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
    c.execute('SELECT filename, upload_time, parsed_data, jd_text, llm_analysis, tokens_used, api_cost FROM uploads WHERE id = ?', (upload_id,))
    result = c.fetchone()
    conn.close()

    if not result:
        return HTMLResponse("未找到记录", status_code=404)

    filename, upload_time, parsed_json, jd_text, llm_analysis_json, tokens_used, api_cost = result
    parsed_data = json.loads(parsed_json) if parsed_json else {}
    llm_analysis = json.loads(llm_analysis_json) if llm_analysis_json else None

    return templates.TemplateResponse("result.html", {
        "request": request,
        "filename": filename,
        "upload_time": upload_time,
        "parsed_data": parsed_data,
        "jd_text": jd_text,
        "llm_analysis": llm_analysis,
        "tokens_used": tokens_used or 0,
        "api_cost": api_cost or 0.0
    })

@app.get("/stats", response_class=HTMLResponse)
async def get_stats(request: Request):
    """查看 API 成本统计"""
    conn = sqlite3.connect('resumes.db')
    c = conn.cursor()

    # 获取所有记录
    c.execute('SELECT id, filename, upload_time, tokens_used, api_cost FROM uploads ORDER BY upload_time DESC')
    records = []
    total_tokens = 0
    total_cost = 0.0

    for row in c.fetchall():
        record_id, filename, upload_time, tokens_used, api_cost = row
        tokens_used = tokens_used or 0
        api_cost = api_cost or 0.0

        records.append({
            'id': record_id,
            'filename': filename,
            'upload_time': upload_time,
            'tokens_used': tokens_used,
            'api_cost': api_cost
        })

        total_tokens += tokens_used
        total_cost += api_cost

    conn.close()

    total_count = len(records)
    avg_cost = total_cost / total_count if total_count > 0 else 0.0

    return templates.TemplateResponse("stats.html", {
        "request": request,
        "records": records,
        "total_count": total_count,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "avg_cost": avg_cost
    })

