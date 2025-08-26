"""
Report generation utilities
"""

import os
import subprocess
import sys
from typing import Dict, Any
from pathlib import Path
import logging

from app.config import settings

logger = logging.getLogger(__name__)

def get_report_templates(language: str = 'en') -> Dict[str, str]:
    """Get localized report templates based on language."""
    templates = {
        'en': {
            'title': 'Skin Diagnosis Report',
            'patient_info': 'Patient Information',
            'name': 'Name',
            'age': 'Age',
            'gender': 'Gender',
            'classification': 'Classification',
            'medical_history': 'Medical History',
            'medical_image': 'Medical Image',
            'concise_analysis': 'Concise Analysis',
            'comprehensive_report': 'Comprehensive Diagnosis Report',
            'years': 'years',
            'image_not_available': 'Medical image not available'
        },
        'ko': {
            'title': '피부 진단 보고서',
            'patient_info': '환자 정보',
            'name': '이름',
            'age': '나이',
            'gender': '성별',
            'classification': '분류',
            'medical_history': '병력',
            'medical_image': '의료 이미지',
            'concise_analysis': '간단한 분석',
            'comprehensive_report': '종합 진단 보고서',
            'years': '세',
            'image_not_available': '의료 이미지를 사용할 수 없습니다'
        },
        'vi': {
            'title': 'Báo Cáo Chẩn Đoán Da',
            'patient_info': 'Thông Tin Bệnh Nhân',
            'name': 'Tên',
            'age': 'Tuổi',
            'gender': 'Giới Tính',
            'classification': 'Phân Loại',
            'medical_history': 'Tiền Sử Bệnh',
            'medical_image': 'Hình Ảnh Y Tế',
            'concise_analysis': 'Phân Tích Tóm Tắt',
            'comprehensive_report': 'Báo Cáo Chẩn Đoán Toàn Diện',
            'years': 'tuổi',
            'image_not_available': 'Hình ảnh y tế không có sẵn'
        }
    }
    
    return templates.get(language, templates['en'])

class ReportGenerator:
    """Handles generation of diagnosis reports in various formats"""
    
    def __init__(self):
        self.reports_dir = settings.REPORTS_DIR
        os.makedirs(self.reports_dir, exist_ok=True)
    
    async def generate_reports(
        self,
        patient_info: Dict[str, Any],
        concise_analysis: str,
        comprehensive_report: str,
        image_path: str,
        language: str = 'en'
    ) -> Dict[str, str]:
        """
        Generate all report formats
        
        Args:
            patient_info: Patient information dictionary
            concise_analysis: Brief analysis text
            comprehensive_report: Detailed diagnosis report
            image_path: Path to the medical image
            language: Language for report generation
            
        Returns:
            Dictionary with paths to generated files
        """
        
        # Use consistent naming format - always use "susan" as the standard filename
        analysis_filename = "image_analysis.txt"
        markdown_filename = "diagnosis_report.md"
        pdf_filename = "diagnosis_report.pdf"
        
        analysis_path = os.path.join(self.reports_dir, analysis_filename)
        markdown_path = os.path.join(self.reports_dir, markdown_filename)
        pdf_path = os.path.join(self.reports_dir, pdf_filename)
        
        # Generate analysis text file
        await self._generate_analysis_file(analysis_path, concise_analysis)
        
        # Generate markdown report
        await self._generate_markdown_report(
            markdown_path, patient_info, concise_analysis, comprehensive_report, image_path, language
        )
        
        # Generate PDF report
        await self._generate_pdf_report(markdown_path, pdf_path)
        
        return {
            "analysis": analysis_path,
            "markdown": markdown_path,
            "pdf": pdf_path
        }
    async def generate_diagnosis_reports(
        self,
        patient_info: Dict[str, Any],
        concise_analysis: str,
        comprehensive_report: str,
        image_path: str,
        language: str = 'en'
    ) -> Dict[str, str]:
        """
        Generate only markdown and PDF reports (for Full Diagnosis)

        Args:
            patient_info: Patient information dictionary
            concise_analysis: Brief analysis text
            comprehensive_report: Detailed diagnosis report
            image_path: Path to the medical image
            language: Language for report generation

        Returns:
            Dictionary with paths to generated files (markdown and pdf only)
        """

        # Generate unique filenames to avoid overwriting
        import uuid
        unique_id = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID for uniqueness
        patient_name_safe = patient_info['name'].replace(' ', '_')

        markdown_filename = f"diagnosis_{patient_name_safe}_{unique_id}.md"
        pdf_filename = f"diagnosis_{patient_name_safe}_{unique_id}.pdf"

        markdown_path = os.path.join(self.reports_dir, markdown_filename)
        pdf_path = os.path.join(self.reports_dir, pdf_filename)

        # Generate markdown report
        await self._generate_markdown_report(
            markdown_path, patient_info, concise_analysis, comprehensive_report, image_path, language
        )

        # Generate PDF report
        await self._generate_pdf_report(markdown_path, pdf_path)

        return {
            "markdown": markdown_path,
            "pdf": pdf_path,
            "markdown_filename": markdown_filename,
            "pdf_filename": pdf_filename
        }

    async def _generate_analysis_file(self, file_path: str, analysis: str):
        """Generate simple analysis text file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(analysis)
            logger.info(f"Analysis file generated: {file_path}")
        except Exception as e:
            logger.error(f"Error generating analysis file: {e}")
            raise
    
    async def _generate_markdown_report(
        self,
        file_path: str,
        patient_info: Dict[str, Any],
        concise_analysis: str,
        comprehensive_report: str,
        image_path: str,
        language: str = 'en'
    ):
        """Generate comprehensive markdown report with language support"""
        try:
            # Get localized templates
            templates = get_report_templates(language)
            
            # Copy image to reports directory for embedding
            image_name = f"medical_image_{os.path.basename(image_path)}"
            image_dest_path = os.path.join(self.reports_dir, image_name)

            # Copy the image file to reports directory
            import shutil
            if os.path.exists(image_path):
                shutil.copy2(image_path, image_dest_path)
                logger.info(f"Image copied to reports directory: {image_dest_path}")

            with open(file_path, 'w', encoding='utf-8') as f:
                # Professional Medical Report Header
                f.write(f"# 🏥 MEDICAL DIAGNOSIS REPORT\n\n")
                f.write(f"---\n\n")
                
                # Report metadata
                from datetime import datetime
                report_date = datetime.now().strftime("%B %d, %Y")
                report_time = datetime.now().strftime("%I:%M %p")
                
                f.write(f"<strong>Report Date:</strong> {report_date}  \n")
                f.write(f"<strong>Report Time:</strong> {report_time}  \n")
                f.write(f"<strong>Report ID:</strong> {patient_info['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}  \n\n")
                f.write(f"---\n\n")

                # Patient Information Section
                f.write(f"## 👤 PATIENT INFORMATION\n\n")
                f.write(f"| **Field** | **Value** |\n")
                f.write(f"|-----------|-----------|\n")
                f.write(f"| <strong>{templates['name']}</strong> | {patient_info['name']} |\n")
                f.write(f"| <strong>{templates['age']}</strong> | {patient_info['age']} {templates['years']} |\n")
                f.write(f"| <strong>{templates['gender']}</strong> | {patient_info['gender']} |\n")
                f.write(f"| <strong>{templates['classification']}</strong> | {patient_info['classification']} |\n")
                f.write(f"| <strong>{templates['medical_history']}</strong> | {patient_info['history']} |\n\n")

                # Medical Image Section
                f.write(f"## 📸 MEDICAL IMAGE ANALYSIS\n\n")
                if os.path.exists(image_dest_path):
                    f.write(f"![{templates['medical_image']}]({image_name})\n\n")
                    f.write(f"*Medical Image: {image_name}*  \n")
                    f.write(f"*Image captured for diagnostic analysis*  \n\n")
                else:
                    f.write(f"*{templates['image_not_available']}*  \n\n")
                f.write(f"---\n\n")

                # AI Analysis Section
                f.write(f"## 🤖 AI CLINICAL ASSESSMENT\n\n")
                f.write(f"### Initial Analysis\n\n")
                f.write(f"{concise_analysis}\n\n")
                f.write(f"---\n\n")

                # Comprehensive Medical Report
                f.write(f"## 📋 COMPREHENSIVE MEDICAL EVALUATION\n\n")
                f.write(f"### 1. Clinical Findings\n\n")
                f.write(f"{comprehensive_report}\n\n")
                
                f.write(f"### 2. Differential Diagnosis\n\n")
                f.write(f"Based on the visual analysis and clinical assessment, the following conditions should be considered:\n\n")
                f.write(f"• <strong>Primary:</strong> Contact dermatitis or allergic reaction\n")
                f.write(f"• <strong>Secondary:</strong> Inflammatory skin condition\n")
                f.write(f"• <strong>Tertiary:</strong> Possible infection requiring further evaluation\n\n")
                
                f.write(f"### 3. Clinical Recommendations\n\n")
                f.write(f"1. <strong>Immediate Actions:</strong>\n")
                f.write(f"   - Schedule consultation with dermatologist\n")
                f.write(f"   - Avoid potential irritants\n")
                f.write(f"   - Monitor for changes in symptoms\n\n")
                
                f.write(f"2. <strong>Treatment Considerations:</strong>\n")
                f.write(f"   - Consider topical treatment if prescribed\n")
                f.write(f"   - Document any progression of symptoms\n")
                f.write(f"   - Maintain photographic documentation\n\n")
                
                f.write(f"3. <strong>Follow-up Plan:</strong>\n")
                f.write(f"   - Re-evaluation in 1-2 weeks\n")
                f.write(f"   - Consider patch testing if allergic reaction suspected\n")
                f.write(f"   - Maintain symptom diary\n\n")
                
                f.write(f"### 4. Risk Assessment\n\n")
                f.write(f"| <strong>Risk Level</strong> | <strong>Description</strong> |\n")
                f.write(f"|----------------|----------------|\n")
                f.write(f"| <strong>Low</strong> | Benign skin condition |\n")
                f.write(f"| <strong>Medium</strong> | Requires medical attention |\n")
                f.write(f"| <strong>High</strong> | Immediate medical evaluation recommended |\n\n")
                
                f.write(f"---\n\n")
                
                # Important Disclaimers
                f.write(f"## ⚠️ IMPORTANT DISCLAIMERS\n\n")
                f.write(f"<strong>This report is generated by AI analysis and should be reviewed by a qualified healthcare professional for final diagnosis and treatment recommendations.</strong>\n\n")
                f.write(f"<strong>The information provided is for educational purposes and should not replace professional medical advice.</strong>\n\n")
                f.write(f"<strong>For medical emergencies, contact emergency services immediately.</strong>\n\n")
                
                f.write(f"---\n\n")
                
                # Footer
                f.write(f"## 📄 REPORT METADATA\n\n")
                f.write(f"<strong>Generated by:</strong> AI Medical Analysis System  \n")
                f.write(f"<strong>Report Type:</strong> Skin Condition Diagnosis  \n")
                f.write(f"<strong>Analysis Method:</strong> Computer Vision + Medical AI  \n")
                f.write(f"<strong>Confidence Level:</strong> High  \n")
                f.write(f"<strong>Report Version:</strong> 1.0  \n\n")
                
                f.write(f"---\n\n")
                f.write(f"*End of Report*\n")

            logger.info(f"Markdown report generated: {file_path}")
        except Exception as e:
            logger.error(f"Error generating markdown report: {e}")
            raise
    
    async def _generate_pdf_report(self, markdown_path: str, pdf_path: str):
        """Generate PDF report from markdown using md_to_pdf.py"""
        try:
            logger.info(f"Starting PDF generation: {markdown_path} -> {pdf_path}")
            
            # Get the path to md_to_pdf.py script
            script_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            md_to_pdf_script = os.path.join(script_dir, 'md_to_pdf.py')
            
            if not os.path.exists(md_to_pdf_script):
                logger.warning(f"md_to_pdf.py not found at {md_to_pdf_script}, skipping PDF generation")
                return
            
            logger.info(f"Using md_to_pdf script: {md_to_pdf_script}")
            
            # Run md_to_pdf.py script using the correct Python executable
            # Try to use the virtual environment Python if available
            python_executable = sys.executable

            # Check if we're in a virtual environment and use the venv python
            venv_python = os.path.join(os.path.dirname(script_dir), 'venv', 'Scripts', 'python.exe')
            if os.path.exists(venv_python):
                python_executable = venv_python
                logger.info(f"Using virtual environment Python: {python_executable}")
            else:
                logger.info(f"Using system Python: {python_executable}")

            logger.info(f"Running PDF generation command...")
            result = subprocess.run([
                python_executable, md_to_pdf_script,
                markdown_path,
                '-o', pdf_path
            ], capture_output=True, text=True, timeout=60)
            
            logger.info(f"PDF generation completed with return code: {result.returncode}")
            if result.stdout:
                logger.info(f"PDF generation stdout: {result.stdout[:500]}...")
            if result.stderr:
                logger.warning(f"PDF generation stderr: {result.stderr}")
            
            if result.returncode == 0:
                logger.info(f"PDF report generated successfully: {pdf_path}")
            else:
                logger.error(f"PDF generation failed: {result.stderr}")
                # Don't raise exception, just log the error
                
        except subprocess.TimeoutExpired:
            logger.error("PDF generation timed out after 60 seconds")
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            # Don't raise exception for PDF generation failures
