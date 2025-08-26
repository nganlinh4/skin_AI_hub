"""
Language-specific prompt templates for VLM diagnosis service
"""

from typing import Dict, Any

class LanguagePromptManager:
    """Manages language-specific prompts for diagnosis service"""
    
    def __init__(self):
        self.language_prompts = {
            "en": {
                "diagnosis_template": """You are an expert dermatologist. Please generate a comprehensive skin diagnosis report based on the provided image and patient information.

Patient Information:
- Name: {patient_name}
- Age: {patient_age} years old
- Gender: {patient_gender}
- Classification: {classification_result}
- Medical History: {skin_history}

Please provide a detailed analysis including:
1. Visual observations of the skin condition
2. Differential diagnosis considerations
3. Recommended treatment approach
4. Follow-up recommendations
5. Patient education points

Format your response in clear sections with appropriate medical terminology while remaining accessible to healthcare professionals.""",
                
                "analysis_template": """Please provide a brief, single-sentence analysis of the skin condition shown in this medical image.

Patient Information:
- Age: {patient_age} years
- Gender: {patient_gender}
- Classification: {classification}

Focus on the most prominent visual features of the skin condition.""",
                
                "system_message_diagnosis": "You are an expert dermatologist with extensive experience in diagnosing skin conditions.",
                
                "system_message_analysis": "You are an expert dermatologist. Provide a concise, single-sentence analysis of the skin condition shown in the image."
            },
            
            "ko": {
                "diagnosis_template": """당신은 전문 피부과 의사입니다. 제공된 이미지와 환자 정보를 바탕으로 포괄적인 피부 진단 보고서를 한국어로 작성해 주세요.

환자 정보:
- 이름: {patient_name}
- 나이: {patient_age}세
- 성별: {patient_gender}
- 분류: {classification_result}
- 병력: {skin_history}

다음 내용을 포함한 상세한 분석을 제공해 주세요:
1. 피부 상태의 시각적 관찰 소견
2. 감별 진단 고려사항
3. 권장 치료 방법
4. 추적 관찰 권장사항
5. 환자 교육 요점

의료진이 이해하기 쉽도록 적절한 의학 용어를 사용하여 명확한 섹션으로 구성해 주세요.""",
                
                "analysis_template": """이 의료 이미지에 나타난 피부 상태에 대한 간단한 한 문장 분석을 한국어로 제공해 주세요.

환자 정보:
- 나이: {patient_age}세
- 성별: {patient_gender}
- 분류: {classification}

피부 상태의 가장 두드러진 시각적 특징에 집중해 주세요.""",
                
                "system_message_diagnosis": "당신은 피부 질환 진단에 풍부한 경험을 가진 전문 피부과 의사입니다. 항상 한국어로 답변해 주세요.",
                
                "system_message_analysis": "당신은 전문 피부과 의사입니다. 이미지에 나타난 피부 상태에 대한 간결하고 정확한 한 문장 분석을 한국어로 제공해 주세요."
            },
            
            "vi": {
                "diagnosis_template": """Bạn là một bác sĩ da liễu chuyên nghiệp. Hãy tạo một báo cáo chẩn đoán da liễu chi tiết bằng tiếng Việt dựa trên hình ảnh và thông tin bệnh nhân được cung cấp.

Thông tin bệnh nhân:
- Tên: {patient_name}
- Tuổi: {patient_age} tuổi
- Giới tính: {patient_gender}
- Phân loại: {classification_result}
- Tiền sử bệnh: {skin_history}

Hãy cung cấp phân tích chi tiết bao gồm:
1. Quan sát trực quan về tình trạng da
2. Các chẩn đoán phân biệt cần xem xét
3. Phương pháp điều trị được khuyến nghị
4. Khuyến nghị theo dõi
5. Các điểm giáo dục bệnh nhân

Hãy trình bày phản hồi theo các phần rõ ràng với thuật ngữ y khoa phù hợp nhưng vẫn dễ hiểu đối với các chuyên gia y tế.""",
                
                "analysis_template": """Hãy cung cấp một phân tích ngắn gọn trong một câu về tình trạng da được hiển thị trong hình ảnh y tế này bằng tiếng Việt.

Thông tin bệnh nhân:
- Tuổi: {patient_age} tuổi
- Giới tính: {patient_gender}
- Phân loại: {classification}

Tập trung vào các đặc điểm trực quan nổi bật nhất của tình trạng da.""",
                
                "system_message_diagnosis": "Bạn là bác sĩ da liễu chuyên nghiệp có nhiều kinh nghiệm trong chẩn đoán các bệnh về da. Luôn trả lời bằng tiếng Việt.",
                
                "system_message_analysis": "Bạn là bác sĩ da liễu chuyên nghiệp. Cung cấp phân tích ngắn gọn và chính xác trong một câu về tình trạng da hiển thị trong hình ảnh bằng tiếng Việt."
            }
        }
    
    def get_prompts(self, language: str = "en") -> Dict[str, str]:
        """Get prompts for specified language"""
        if language not in self.language_prompts:
            language = "en"  # Default to English
        return self.language_prompts[language]
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages"""
        return list(self.language_prompts.keys())
    
    def format_diagnosis_prompt(self, language: str, patient_info: Dict[str, Any]) -> str:
        """Format diagnosis prompt with patient information"""
        prompts = self.get_prompts(language)
        template = prompts["diagnosis_template"]
        
        return template.format(
            patient_name=patient_info.get('name', 'Patient'),
            patient_age=patient_info.get('age', 'Unknown'),
            patient_gender=patient_info.get('gender', 'Unknown'),
            classification_result=patient_info.get('classification', 'Unknown'),
            skin_history=patient_info.get('history', 'No medical history provided')
        )
    
    def format_analysis_prompt(self, language: str, patient_info: Dict[str, Any]) -> str:
        """Format analysis prompt with patient information"""
        prompts = self.get_prompts(language)
        template = prompts["analysis_template"]
        
        return template.format(
            patient_age=patient_info.get('age', 'Unknown'),
            patient_gender=patient_info.get('gender', 'Unknown'),
            classification=patient_info.get('classification', 'Unknown')
        )
