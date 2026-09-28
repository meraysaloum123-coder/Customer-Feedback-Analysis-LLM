import pandas as pd
from transformers import pipeline

# 1. قراءة البيانات
df = pd.read_csv('customer_feedback.csv')

# 2. حساب الإحصائيات
meanR = df['Rating'].mean()
positive_count = len(df[df['Rating'] >= 4])
negative_count = len(df[df['Rating'] <= 2])
neutral_count = len(df[df['Rating'] == 3])

# استخراج أكثر 25 تعليقاً سلبياً تكراراً (الخيار الثاني)
negative_comments = df[df['Rating'] <= 2]['Comment'].value_counts().head(25).index.tolist()

# 3. إعداد الـ Prompt
system_instruction = "You are a professional strategic data analyst. Your task is to transform raw customer feedback data and statistics into a concise, actionable executive report."

user_query = f"""
Based on the following customer feedback data, please write an analytical report:

[Statistics]:
- Average Rating: {meanR:.2f}
- Positive Ratings (>= 4): {positive_count}
- Negative Ratings (<= 2): {negative_count}
- Neutral Ratings (= 3): {neutral_count}

[Sample of Negative Feedback for Analysis]:
{negative_comments}

Requirements for the report:
1. Assess the overall customer satisfaction level (Low/Medium/High) and justify it using the provided statistics.
2. Identify the top key patterns or recurring issues found in the negative comments above.
   - IMPORTANT: Connect specific negative comments to their low ratings to identify the root causes.
   - IMPORTANT: Focus only on qualitative analysis for patterns. Do not associate specific count numbers from the [Statistics] section with these qualitative patterns.
3. Propose two practical, actionable suggestions to improve the service and increase customer satisfaction.

Please write the report in a professional tone, structured with clear bullet points.
"""

messages = [
    {"role": "system", "content": system_instruction},
    {"role": "user", "content": user_query}
]

# 4. إعداد الـ Pipeline
pipe = pipeline("text-generation", model="google/gemma-3-1b-it")

# 5. توليد التقرير
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
response = pipe(prompt, max_new_tokens=800) # زيادة عدد التوكنز قليلاً لاستيعاب التقرير

# استخراج نص التقرير من المخرجات
# ملاحظة: النموذج يرجع كامل الـ prompt مع الرد، سنأخذ ما بعد الـ prompt فقط
full_text = response[0]['generated_text']
report_text = full_text.split("<start_of_turn>model")[-1].strip()

print(report_text)

# 6. تصدير التقرير إلى ملف نصي
report_filename = "Executive_Feedback_Report.md"
with open(report_filename, "w", encoding="utf-8") as file:
    file.write(report_text)

print(f"\nReport successfully exported to: {report_filename}")
