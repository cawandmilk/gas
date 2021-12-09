import re

from typing import List


class CleanNewspaperArticleBase():

    def __init__(
        self, 
    ):
        super(CleanNewspaperArticleBase, self).__init__()


    @staticmethod
    def remove_reporter_info(article_original: List[str]) -> List[str]:
        email_pattern = re.compile(r"[a-zA-Z0-9+-_.]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
        reporter_pattern = re.compile(r"^\/?([가-힣]+)\s?(기자|팀)?\s*\.?$")

        texts = []
        reporters = []
        for sentence in article_original:
            ## If we can find email pattern in one sentence:
            email = email_pattern.findall(sentence)
            if email != []:
                continue
            
            ## We don't care about "OOO 기자 OOO 기자", not "OOO 기자"
            reporter = reporter_pattern.findall(sentence)
            if reporter != []:
                reporters.extend([i[0] for i in reporter])
                continue

            ## If known reporter name is in sentence...
            if any([i in sentence for i in reporters]):
                continue
            
            texts.append(sentence)
        
        return texts


    @staticmethod
    def remove_url(article_original: List[str]) -> List[str]:
        ## (e.g. id="357606465")
        ## Ref: https://www.codegrepper.com/code-examples/python/regex+for+url+python
        url_pattern = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", re.MULTILINE)

        texts = "\n".join(article_original)
        texts = url_pattern.sub("", texts)

        return texts.split("\n")


    @staticmethod
    def remove_phone_number(article_original: List[str]) -> List[str]:
        phone_pattern = re.compile(r"""(
            (\d{2}|\(\d{2}\)|\d{3}|\(\d{3}\))?      ## 2 or 3 words include "(...)" patterns -> optional
            (|-|\.)?                                ## sep word: "." or "-"
            (\d{3}|\d{4})                           ## 3 or 4 numbers
            (\s|-|\.)                               ## sep word: "." or "-"
            (\d{4})                                 ## 4 numbers
        )""", re.VERBOSE | re.MULTILINE)  

        texts = "\n".join(article_original)
        texts = phone_pattern.sub("", texts)

        return texts.split("\n")


    @staticmethod
    def remove_brack_sentence(article_original: List[str]) -> List[str]:
        bracket_1_pattern = re.compile(r"\s?<.*>", re.MULTILINE)        ## e.g. "<OOO씨 제공>"
        bracket_2_pattern = re.compile(r"\s?\(.*\)", re.MULTILINE)      ## e.g. 
        bracket_3_pattern = re.compile(r"\s?\[.*\]", re.MULTILINE)      ## e.g. 

        texts = "\n".join(article_original)
        texts = bracket_1_pattern.sub("", texts)
        texts = bracket_2_pattern.sub("", texts)
        texts = bracket_3_pattern.sub("", texts)
        
        return texts.split("\n")


    @staticmethod
    def remove_supplementary_sentence(article_original: List[str]) -> List[str]:
        ## e.g. "/광주시양궁협회 제공"
        ## But, it may be an important sentence... (e.g. id="334957827")
        supplementary_sentence_pattern = re.compile(r"^\s?[▶|\/][.]*", re.MULTILINE)

        texts = "\n".join(article_original)
        texts = supplementary_sentence_pattern.sub("", texts)
        
        return texts.split("\n")


    @staticmethod
    def replace_start_with_hyphen(article_original: List[str]) -> List[str]:
        ## e.g. 
        hyphen_pattern = re.compile(r"^\s?\-", re.MULTILINE)

        texts = "\n".join(article_original)
        texts = hyphen_pattern.sub("", texts)

        return texts.split("\n")

    
    @staticmethod
    def replace_universal_apostrophe(article_original: List[str]) -> List[str]:
        ## “ (U+201C), ” (U+201D) -> " (U+0022)
        ## ‘ (U+2018), ’ (U+2019) -> ' (U+0027)
        small_apostrophe_pattern = re.compile(r"[‘|’]", re.MULTILINE)
        large_apostrophe_pattern = re.compile(r"[“|”]", re.MULTILINE)

        texts = "\n".join(article_original)
        texts = small_apostrophe_pattern.sub("'", texts)
        texts = large_apostrophe_pattern.sub("\"", texts)

        return texts.split("\n")


    @staticmethod
    def replace_repeated_apostrophe(article_original: List[str]) -> List[str]:
        ## (e.g. id="330644133")
        small_apostrophe_pattern = re.compile(r"'{2,}", re.MULTILINE)
        large_apostrophe_pattern = re.compile(r"\"{2,}", re.MULTILINE)

        texts = "\n".join(article_original)
        texts = small_apostrophe_pattern.sub("'", texts)
        texts = large_apostrophe_pattern.sub("\"", texts)

        return texts.split("\n")


    @staticmethod
    def apply(article_original: List[str]) -> List[str]:
        texts = article_original
        
        ## Remove functions.
        texts = CleanNewspaperArticleBase.remove_reporter_info(texts)
        texts = CleanNewspaperArticleBase.remove_url(texts)
        texts = CleanNewspaperArticleBase.remove_phone_number(texts)
        # texts = CleanNewspaperArticleBase.remove_supplementary_sentence(texts)
        # texts = CleanNewspaperArticleBase.remove_brack_sentence(texts)

        ## Replace functions.
        # texts = CleanNewspaperArticleBase.replace_start_with_hyphen(texts)
        # texts = CleanNewspaperArticleBase.replace_universal_apostrophe(texts)
        # texts = CleanNewspaperArticleBase.replace_repeated_apostrophe(texts)

        return texts


class CleanNewspaperArticle():
    
    def __init__(
        self,
    ):
        super(CleanNewspaperArticle, self).__init__()
        self.media_name_to_function = {
            "건설경제": self._cnews,
            "광양신문": self._gynet,
            "광주매일신문": self._kjdaily,
            "광주일보": self._kwangju,
            "국제신문": self._kookje,
            "기호일보": self._kihoilbo,
            "남도일보": self._namdonews,
            "당진시대": self._djtimes,
            "대구신문": self._idaegu_co_kr,
            "대구일보": self._idaegu_com,
            "대전일보": self._daejonilbo,
            "동양일보": self._dynews,
            "디지털타임스": self._dt,
            "매일경제": self._mk,
            "매일신문": self._imaeil,
            "머니투데이": self._mt,
            "무등일보": self._honam,
            "부산일보": self._busan,
            "새전북신문": self._sjbnews,
            "서울경제": self._sedaily,
            "서울신문": self._seoul,
            "아시아경제": self._asiae,
            "아주경제": self._ajunews,
            "영남일보": self._yeongnam,
            "울산매일": self._iusm,
            "이데일리": self._edaily,
            "인천일보": self._incheonilbo,
            "전남일보": self._jnilbo,
            "전라일보": self._jeollailbo,
            "전북도민일보": self._domin,
            "전북일보": self._jjan,
            "제민일보": self._jemin,
            "제주일보": self._jejunews,
            "중도일보": self._joongdo,
            "중부매일": self._jbnews,
            "중부일보": self._joongboo,
            "충북일보": self._inews365,
            "충청일보": self._ccdailynews,
            "충청투데이": self._cctoday,
            "한국경제": self._hankyung,
            "한라일보": self._ihalla,
            "환경일보": self._hkbs,
        }


    def _cnews(self, article_original: List[str]) -> List[str]:
        """ 건설경제 """
        return article_original


    def _gynet(self, article_original: List[str]) -> List[str]:
        """ 광양신문 """
        return article_original


    def _kjdaily(self, article_original: List[str]) -> List[str]:
        """ 광주매일신문 """
        return article_original


    def _kwangju(self, article_original: List[str]) -> List[str]:
        """ 광주일보 """
        ## Remove standalone "출처 :" or "출처:"
        reference_pattern = re.compile(r"^(출처)\s?\:", re.MULTILINE)

        texts = "\n".join(article_original)
        texts = reference_pattern.sub("", texts)

        return texts.split("\n")


    def _kookje(self, article_original: List[str]) -> List[str]:
        """ 국제신문 """
        sponser_pattern = re.compile(r"제공\s?$", re.MULTILINE)
        dotted_pattern = re.compile(r"^\.$", re.MULTILINE)

        texts = "\n".join(article_original)
        texts = sponser_pattern.sub("", texts)
        texts = dotted_pattern.sub("", texts)

        return texts.split("\n")


    def _kihoilbo(self, article_original: List[str]) -> List[str]:
        """ 기호일보 """
        ## Media depended style:
        ##   e.g. "양주=OOO 기자 OOO@OOO.co.kr"
        ##   e.g. "기호일보, KIHOILBO"
        reporter_pattern = re.compile(r"^[가-힣]+\=[가-힣]+\s?(기자|팀)?$", re.MULTILINE)
        media_pattern = re.compile(r"^\s?기호일보, KIHOILBO\s?$", re.MULTILINE)
        picture_reference_pattern = re.compile(r"^\s?(포토|사진)\s?\:[.]*", re.MULTILINE)

        texts = "\n".join(article_original)
        texts = reporter_pattern.sub("", texts)
        texts = media_pattern.sub("", texts)
        texts = picture_reference_pattern.sub("", texts)
        
        return texts.split("\n")


    def _namdonews(self, article_original: List[str]) -> List[str]:
        """ 남도일보 """
        picture_reference_pattern = re.compile(r"^\s?사진\s?\=[.]*", re.MULTILINE)
        
        texts = "\n".join(article_original)
        texts = picture_reference_pattern.sub("", texts)

        return texts.split("\n")


    def _djtimes(self, article_original: List[str]) -> List[str]:
        """ 당진시대 """
        removed_special_token_pattern = re.compile(r"^\s?(▲|■|※)[.]*", re.MULTILINE)
        replaced_special_token_pattern = re.compile(r"^\s?[>]{2}", re.MULTILINE)

        texts = "\n".join(article_original)
        texts = removed_special_token_pattern.sub("", texts)
        texts = replaced_special_token_pattern.sub("", texts)

        return texts.split("\n")


    def _idaegu_co_kr(self, article_original: List[str]) -> List[str]:
        """ 대구신문 """
        return article_original


    def _idaegu_com(self, article_original: List[str]) -> List[str]:
        """ 대구일보 """
        return article_original


    def _daejonilbo(self, article_original: List[str]) -> List[str]:
        """ 대전일보 """
        ## e.g. "한국관광 100선에 선정된 궁남지 전경 사진=부여군 제공 [부여]"
        ## e.g. "사진=OOO 의원 제공"
        ## It will not care s.t. have no "제공" in the end of reference.

        # reference_pattern = re.compile(r"(사진)\s?\=\s?([가-힣]*\s?)+(제공)", re.MULTILINE)

        # texts = "\n".join(article_original)
        # texts = reference_pattern.sub("", texts)
        # print("Done")

        # return texts.split("\n")

        """ Skip -> cause error because of too many iteration steps... (too high complexity) """
        return article_original


    def _dynews(self, article_original: List[str]) -> List[str]:
        """ 동양일보 """
        ## e.g. "(동양일보 OOO 기자)"
        company_mark_pattern = re.compile("동양일보")

        texts = "\n".join(article_original)
        texts = company_mark_pattern.sub("", texts)

        return texts.split("\n")


    def _dt(self, article_original: List[str]) -> List[str]:
        """ 디지털타임스 """
        replaced_special_token_pattern = re.compile(r"^\s?◇", re.MULTILINE)

        texts = "\n".join(article_original)
        texts = replaced_special_token_pattern.sub("", texts)

        return texts.split("\n")


    def _mk(self, article_original: List[str]) -> List[str]:
        """ 매일경제 """
        ## (e.g. id="354419753")
        advertisement_pattern = re.compile(r"스탁론", re.MULTILINE)

        texts = "\n".join(article_original)
        ## If you can find advertisement documents...
        if advertisement_pattern.findall(texts) != []:
            ## Skip it.
            return [""]
        
        return article_original
        

    def _imaeil(self, article_original: List[str]) -> List[str]:
        """ 매일신문 """
        return article_original


    def _mt(self, article_original: List[str]) -> List[str]:
        """ 머니투데이 """
        return article_original


    def _honam(self, article_original: List[str]) -> List[str]:
        """ 무등일보 """
        return article_original


    def _busan(self, article_original: List[str]) -> List[str]:
        """ 부산일보 """
        ## (e.g. id="350912775")
        ## Reporter's email will represented as "OOO 기자 OOO@" (ends with "@")
        email_pattern = re.compile(r"[\S]*@$", re.MULTILINE)

        texts = "\n".join(article_original)
        texts = email_pattern.sub("", texts)

        ## (e.g. id="359520459")
        ## Advertisements pattern.
        advertisement_pattern_1 = re.compile(r"^▶ 네이버에서 부산일보 구독하기 클릭!$", re.MULTILINE)
        advertisement_pattern_2 = re.compile(r"^▶ 부산일보 구독하고 스타벅스 Get 하자!$", re.MULTILINE)
        advertisement_pattern_3 = re.compile(r"^▶ 부산일보 홈 바로가기$", re.MULTILINE)

        texts = advertisement_pattern_1.sub("", texts)
        texts = advertisement_pattern_2.sub("", texts)
        texts = advertisement_pattern_3.sub("", texts)

        return texts.split("\n")


    def _sjbnews(self, article_original: List[str]) -> List[str]:
        """ 새전북신문 """
        return article_original


    def _sedaily(self, article_original: List[str]) -> List[str]:
        """ 서울경제 """
        return article_original


    def _seoul(self, article_original: List[str]) -> List[str]:
        """ 서울신문 """
        return article_original


    def _asiae(self, article_original: List[str]) -> List[str]:
        """ 아시아경제 """
        return article_original


    def _ajunews(self, article_original: List[str]) -> List[str]:
        """ 아주경제 """
        return article_original


    def _yeongnam(self, article_original: List[str]) -> List[str]:
        """ 영남일보 """
        return article_original


    def _iusm(self, article_original: List[str]) -> List[str]:
        """ 울산매일 """
        ## (e.g. id="340995643")
        tag_pattern = re.compile(r"&lt;br[\/]?&gt", re.MULTILINE)

        texts = "\n".join(article_original)
        texts = tag_pattern.sub(" ", texts)

        ## (e.g. id="341038283")
        ref_pattern = re.compile(r"^노컷뉴스$", re.MULTILINE)
        texts = ref_pattern.sub("", texts)

        return texts.split("\n")


    def _edaily(self, article_original: List[str]) -> List[str]:
        """ 이데일리 """
        ## (e.g. id="329454903")
        advertisement_patterns = [
            r"^네이버에서 이데일리 \[구독하기▶\]$",
            r"^빡침해소! 청춘뉘우스~ \[스냅타임▶\]$",
            r"^이데일리 채널 구독하면 \[방탄소년단 실물영접 기회가▶\]$",
            r"^꿀잼가득 \[영상보기▶\] , 빡침해소!$",
            r"^청춘뉘우스~ \[스냅타임▶\]$"
        ]

        texts = "\n".join(article_original)
        for pattern in advertisement_patterns:
            texts = re.sub(pattern, "", texts)

        return texts.split("\n")


    def _incheonilbo(self, article_original: List[str]) -> List[str]:
        """ 인천일보 """
        return article_original


    def _jnilbo(self, article_original: List[str]) -> List[str]:
        """ 전남일보 """
        repeated_word_patterns = [
            r"뉴시스",
            r"편집에디터",
        ]

        texts = "\n".join(article_original)
        for pattern in repeated_word_patterns:
            texts = re.sub(pattern, "", texts)

        return texts.split("\n")


    def _jeollailbo(self, article_original: List[str]) -> List[str]:
        """ 전라일보 """
        ## (e.g. id="329500477")
        email_pattern = re.compile(r"[\S]*@$", re.MULTILINE)

        texts = "\n".join(article_original)
        texts = email_pattern.sub("", texts)
        
        return texts.split("\n")


    def _domin(self, article_original: List[str]) -> List[str]:
        """ 전북도민일보 """
        return article_original


    def _jjan(self, article_original: List[str]) -> List[str]:
        """ 전북일보 """
        repeated_word_patterns = re.compile(r"^전북일보$", re.MULTILINE)

        texts = "\n".join(article_original)
        texts = repeated_word_patterns.sub("", texts)
        
        return texts.split("\n")


    def _jemin(self, article_original: List[str]) -> List[str]:
        """ 제민일보 """
        return article_original


    def _jejunews(self, article_original: List[str]) -> List[str]:
        """ 제주일보 """
        repeated_word_patterns = re.compile(r"^제주신보$", re.MULTILINE)

        texts = "\n".join(article_original)
        texts = repeated_word_patterns.sub("", texts)
        
        return texts.split("\n")


    def _joongdo(self, article_original: List[str]) -> List[str]:
        """ 중도일보 """
        ## (e.g. id="350886753")
        email_pattern = re.compile(r"[\S]*@$", re.MULTILINE)

        texts = "\n".join(article_original)
        texts = email_pattern.sub("", texts)
        
        return texts.split("\n")


    def _jbnews(self, article_original: List[str]) -> List[str]:
        """ 중부매일 """
        return article_original


    def _joongboo(self, article_original: List[str]) -> List[str]:
        """ 중부일보 """
        ## (e.g. id="330572743")
        repeated_word_patterns = re.compile(r"^연합$", re.MULTILINE)

        texts = "\n".join(article_original)
        texts = repeated_word_patterns.sub("", texts)
        
        return texts.split("\n")


    def _inews365(self, article_original: List[str]) -> List[str]:
        """ 충북일보 """
        ## (e.g. id="331218728")
        ## We don't care such like "충북일보=충주]" (id="333262510")
        repeated_word_patterns = re.compile(r"^충북일보\]", re.MULTILINE)

        texts = "\n".join(article_original)
        texts = repeated_word_patterns.sub("", texts)
        
        return texts.split("\n")


    def _ccdailynews(self, article_original: List[str]) -> List[str]:
        """ 충청일보 """
        repeated_word_patterns = re.compile(r"온라인충청일보", re.MULTILINE)

        texts = "\n".join(article_original)
        texts = repeated_word_patterns.sub("", texts)
        
        return texts.split("\n")


    def _cctoday(self, article_original: List[str]) -> List[str]:
        """ 충청투데이 """
        return article_original


    def _hankyung(self, article_original: List[str]) -> List[str]:
        """ 한국경제 """
        ## (e.g. id="335987391")
        repeated_word_patterns = re.compile(r"^\]", re.MULTILINE)

        texts = "\n".join(article_original)
        texts = repeated_word_patterns.sub("", texts)
        
        return texts.split("\n")


    def _ihalla(self, article_original: List[str]) -> List[str]:
        """ 한라일보 """
        return article_original


    def _hkbs(self, article_original: List[str]) -> List[str]:
        """ 환경일보 """
        removed_special_token_pattern = re.compile(r"^=", re.MULTILINE)

        texts = "\n".join(article_original)
        texts = removed_special_token_pattern.sub("", texts)
        
        return texts.split("\n")


    def __call__(self, article_original: List[str], media_name: str) -> List[str]:
        texts = article_original

        ## We cannot be sure that the distribution of the training data domain set and
        ## the validation & test data domain sets are the same.
        ## We should be able to handle even the first seen "media_name". :(
        # func = self.media_name_to_function.get(media_name)
        # if func != None:
        #     texts = func(texts)

        ## Apply common regex after customized cleaner.
        texts = [i.strip() for i in texts if i.strip() != ""]
        texts = CleanNewspaperArticleBase.apply(texts)
        texts = [i.strip() for i in texts if i.strip() != ""]
        return texts
