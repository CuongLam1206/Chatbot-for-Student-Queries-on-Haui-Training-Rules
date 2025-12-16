"""
Query Normalization - Xử lý từ lóng và viết tắt
Chuẩn hóa query từ sinh viên sang ngôn ngữ chuẩn trước khi xử lý
"""

import re
from typing import Dict, List, Tuple


class QueryNormalizer:
    """Chuẩn hóa query để hiểu từ lóng và viết tắt"""
    
    # Từ điển viết tắt phổ biến
    ABBREVIATIONS = {
        # Viết tắt chung
        "sv": "sinh viên",
        "gv": "giảng viên",
        "cb": "cán bộ",
        "hs": "học sinh",
        
        # Đào tạo
        "đktc": "đăng ký tín chỉ",
        "đkhp": "đăng ký học phần",
        "tc": "tín chỉ",
        "hp": "học phần",
        "hk": "học kỳ",
        "ctđt": "chương trình đào tạo",
        "tn": "tốt nghiệp",
        "xltn": "xét tốt nghiệp",
        "bv": "bảo vệ",
        "kltn": "khóa luận tốt nghiệp",
        "đatn": "đồ án tốt nghiệp",
        
        # Điểm số
        "dtb": "điểm trung bình",
        "đtbhk": "điểm trung bình học kỳ",
        "đtbtl": "điểm trung bình tích lũy",
        "gpa": "grade point average",
        "cpa": "cumulative point average",
        
        # Thủ tục
        "đđ": "đăng ký",
        "đk": "đăng ký",
        "nv": "nhà vệ sinh",  # joke, remove this
        "ktx": "ký túc xá",
        "bhyt": "bảo hiểm y tế",
        "bhtn": "bảo hiểm thất nghiệp",
        
        # Khác
        "qc": "quy chế",
        "qđ": "quyết định",
        "cv": "công văn",
        "tb": "thông báo",
    }
    
    # Từ lóng sinh viên
    SLANG_TERMS = {
        # Học tập
        "rớt môn": "điểm f",
        "trượt môn": "điểm f",
        "trượt": "không đạt",
        "pass": "đạt",
        "đậu": "đạt",
        "ăn điểm": "học lại",
        "học lại": "đăng ký học lại",
        "cày cuốc": "học tập chăm chỉ",
        "cày": "học chăm",
        "gà": "điểm thấp",
        "gà mờ": "điểm kém",
        
        # Điểm số
        "điểm khủng": "điểm cao",
        "điểm cao": "điểm a",
        "điểm giỏi": "điểm a",
        "điểm khá": "điểm b",
        "điểm tb": "điểm c",
        "điểm yếu": "điểm d",
        "điểm kém": "điểm f",
        "bay màu": "điểm f",
        "toang": "điểm f",
        
        # Thủ tục
        "đăng ký môn": "đăng ký học phần",
        "đk môn": "đăng ký học phần",
        "rút môn": "rút bớt học phần",
        "bỏ môn": "rút bớt học phần",
        "nghỉ học": "bảo lưu",
        "nghỉ tạm": "bảo lưu tạm thời",
        "xin nghỉ": "đơn xin nghỉ học",
        
        # Tốt nghiệp
        "ra trường": "tốt nghiệp",
        "tốt nghiệp": "hoàn thành chương trình",
        "nhận bằng": "cấp bằng tốt nghiệp",
        "bảo vệ": "bảo vệ khóa luận",
        "bv đồ án": "bảo vệ đồ án tốt nghiệp",
        
        # Khác
        "thầy": "giảng viên",
        "cô": "giảng viên",
        "phòng đào tạo": "phòng quản lý đào tạo",
        "văn phòng khoa": "phòng đào tạo",
    }
    
    # Các cụm từ đồng nghĩa
    SYNONYMS = {
        "điều kiện tốt nghiệp": ["xét tốt nghiệp", "ra trường", "nhận bằng"],
        "đăng ký học phần": ["đăng ký môn", "đk môn", "đktc", "đkhp"],
        "điểm f": ["rớt môn", "trượt", "không đạt", "bay màu", "toang"],
        "học lại": ["ăn điểm", "thi lại", "đăng ký lại"],
        "bảo lưu": ["nghỉ học", "tạm nghỉ", "nghỉ tạm thời"],
    }
    
    def __init__(self):
        """Initialize normalizer"""
        # Combine all mappings
        self.normalization_map = {
            **self.ABBREVIATIONS,
            **self.SLANG_TERMS
        }
        
        # Build regex pattern for efficient replacement
        self._build_pattern()
    
    def _build_pattern(self):
        """Build regex pattern from all terms"""
        # Sort by length (longest first) to match longer phrases first
        terms = sorted(self.normalization_map.keys(), key=len, reverse=True)
        
        # Escape special regex characters and join with |
        pattern = '|'.join(re.escape(term) for term in terms)
        self.pattern = re.compile(r'\b(' + pattern + r')\b', re.IGNORECASE)
    
    def normalize(self, query: str) -> str:
        """
        Chuẩn hóa query bằng cách thay thế từ lóng/viết tắt
        
        Args:
            query: Câu hỏi gốc từ user
            
        Returns:
            Câu hỏi đã được chuẩn hóa
        """
        normalized = query
        
        # Replace using regex for case-insensitive matching
        def replace_func(match):
            matched_text = match.group(1)
            # Find the key in normalization_map (case-insensitive)
            for key, value in self.normalization_map.items():
                if key.lower() == matched_text.lower():
                    return value
            return matched_text
        
        normalized = self.pattern.sub(replace_func, normalized)
        
        return normalized
    
    def get_explanation(self, query: str) -> List[Tuple[str, str]]:
        """
        Trả về list các từ đã được normalize và ý nghĩa
        
        Args:
            query: Câu hỏi gốc
            
        Returns:
            List of (original_term, normalized_term) tuples
        """
        explanations = []
        
        for match in self.pattern.finditer(query):
            matched_text = match.group(1)
            for key, value in self.normalization_map.items():
                if key.lower() == matched_text.lower():
                    explanations.append((matched_text, value))
                    break
        
        return explanations
    
    def add_custom_term(self, slang: str, standard: str):
        """
        Thêm từ lóng/viết tắt custom
        
        Args:
            slang: Từ lóng hoặc viết tắt
            standard: Từ chuẩn
        """
        self.normalization_map[slang.lower()] = standard.lower()
        self._build_pattern()


# Global normalizer instance
normalizer = QueryNormalizer()


if __name__ == "__main__":
    # Test
    test_queries = [
        "sv rớt môn phải làm gì",
        "đktc như thế nào",
        "điều kiện tn là gì",
        "tôi bị bay màu 3 môn, học lại được không",
        "gpa thấp có ra trường được không",
        "đk môn cho hk sau",
        "cày cuốc cả kỳ vẫn gà",
        "thầy cho điểm f, tôi phải ăn điểm à",
        "bv kltn cần gì"
    ]
    
    print("🔧 Testing Query Normalizer\n")
    print("="*60)
    
    for query in test_queries:
        normalized = normalizer.normalize(query)
        explanations = normalizer.get_explanation(query)
        
        print(f"\n📝 Original:   {query}")
        print(f"✅ Normalized: {normalized}")
        
        if explanations:
            print(f"📖 Terms normalized:")
            for original, standard in explanations:
                print(f"   • '{original}' → '{standard}'")
        
        print("-"*60)
