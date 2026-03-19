#!/usr/bin/env python3
"""
Generate a sample PDF for testing the PDF RAG agent.

Creates a multi-page PDF with varied content including:
- Abstract and introduction
- Methodology section
- Results and conclusions
- Tables (simulated)
"""

import os


def create_sample_pdf():
    """Create a sample PDF document for testing."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("PyMuPDF required. Install with: pip install PyMuPDF")
        return

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sample.pdf")

    doc = fitz.open()

    # ── Page 1: Title & Abstract ──
    page = doc.new_page()
    text = """Impact of AI-Driven Automation on Enterprise Workflows:
A Multi-Organization Study

Authors: Dr. Jane Smith, Prof. John Doe
Published: 2024

Abstract

This paper presents a comprehensive study examining the impact of artificial 
intelligence-driven automation on enterprise workflows across multiple organizations. 
We evaluate productivity improvements, cost reductions, and employee satisfaction 
metrics in three distinct enterprise environments. Our findings demonstrate that 
AI automation leads to significant productivity gains of 25-40% while maintaining 
or improving quality of outputs. The study employs a mixed-methods approach 
combining quantitative metrics with qualitative assessments.

Keywords: artificial intelligence, automation, enterprise workflows, productivity, 
digital transformation
"""
    page.insert_text((72, 72), text, fontsize=11)

    # ── Page 2: Introduction & Related Work ──
    page = doc.new_page()
    text = """1. Introduction

The rapid advancement of artificial intelligence technologies has created 
unprecedented opportunities for enterprise automation. Organizations across 
industries are increasingly adopting AI-driven solutions to streamline workflows, 
reduce manual tasks, and improve decision-making processes.

Traditional workflow management systems relied on rule-based automation, which 
required extensive manual configuration and lacked adaptability. In contrast, 
modern AI systems leverage machine learning models to understand context, make 
predictions, and adapt to changing requirements.

This study addresses three key research questions:
RQ1: What is the quantitative impact of AI automation on productivity?
RQ2: How does AI automation affect employee satisfaction?
RQ3: What are the key factors for successful AI automation adoption?

2. Related Work

Previous studies have examined automation in enterprise contexts. Taylor et al. 
(2022) found that RPA implementations led to 20% efficiency gains. However, 
their study was limited to single organizations. Martinez and Chen (2023) 
explored ML-based document processing, reporting 35% reduction in processing 
time. Our work extends these findings across multiple organizations and 
workflow types.
"""
    page.insert_text((72, 72), text, fontsize=11)

    # ── Page 3: Methodology ──
    page = doc.new_page()
    text = """3. Methodology

3.1 Study Design

We employed a mixed-methods research design combining case studies with 
experimental evaluations across three enterprise environments:

Organization A: Financial services firm (5,000 employees)
Organization B: Healthcare provider (12,000 employees)
Organization C: Manufacturing company (8,000 employees)

3.2 Data Collection

Data was collected over a 12-month period (January 2023 - December 2023) using:
- Automated workflow metrics (processing time, error rates, throughput)
- Employee surveys (N=450, response rate: 78%)
- Semi-structured interviews (N=36)
- System logs and performance data

3.3 AI Systems Evaluated

We evaluated three categories of AI automation:
1. Document processing (OCR + NLP-based classification)
2. Decision support systems (ML-based recommendations)
3. Workflow orchestration (Intelligent process automation)

3.4 Analysis Methods

Quantitative data was analyzed using paired t-tests and ANOVA. Qualitative 
data from interviews was analyzed using thematic analysis following Braun 
and Clarke's (2006) six-phase approach. Statistical significance was set 
at p < 0.05.
"""
    page.insert_text((72, 72), text, fontsize=11)

    # ── Page 4: Results ──
    page = doc.new_page()
    text = """4. Results

4.1 Productivity Impact (RQ1)

AI automation led to significant productivity improvements across all 
three organizations:

Organization A: 38% reduction in document processing time (p < 0.001)
Organization B: 25% improvement in patient record handling (p < 0.01)
Organization C: 42% reduction in quality inspection time (p < 0.001)

Average across organizations: 35% productivity improvement

4.2 Employee Satisfaction (RQ2)

Employee satisfaction scores improved from a mean of 3.2/5.0 (pre-automation) 
to 4.1/5.0 (post-automation). Key drivers included:
- Reduction in repetitive tasks (cited by 85% of respondents)
- More time for creative work (cited by 72%)
- Better decision-making tools (cited by 68%)

4.3 Success Factors (RQ3)

Thematic analysis identified five key success factors:
1. Executive sponsorship and clear vision
2. Phased implementation approach
3. Employee training and change management
4. Integration with existing systems
5. Continuous monitoring and optimization
"""
    page.insert_text((72, 72), text, fontsize=11)

    # ── Page 5: Discussion & Conclusions ──
    page = doc.new_page()
    text = """5. Discussion

Our findings confirm and extend previous research on AI automation benefits. 
The 35% average productivity improvement exceeds the 20% reported by Taylor 
et al. (2022) for RPA-only implementations, suggesting that ML-augmented 
automation provides additional value beyond simple task automation.

The high employee satisfaction scores (4.1/5.0) counter common concerns 
about AI displacing workers. Instead, employees reported feeling more valued 
as they could focus on higher-order tasks.

5.1 Limitations

This study has several limitations:
- Sample limited to three organizations
- 12-month observation period may not capture long-term effects
- Self-reported satisfaction data may have social desirability bias

6. Conclusions

AI-driven automation significantly enhances enterprise workflow efficiency, 
with an average 35% productivity improvement across diverse organizational 
contexts. Successful implementation requires executive sponsorship, phased 
deployment, and strong change management. Future research should examine 
long-term effects and expand to additional industries.

References

Braun, V., & Clarke, V. (2006). Using thematic analysis. Qual. Research, 3(2).
Martinez, R., & Chen, L. (2023). ML Document Processing. AI Review, 15(4).
Taylor, S., et al. (2022). RPA in Enterprise. Business Automation J., 8(1).
"""
    page.insert_text((72, 72), text, fontsize=11)

    doc.save(output_path)
    doc.close()
    print(f"Sample PDF created: {output_path}")
    return output_path


if __name__ == "__main__":
    create_sample_pdf()
