import re

from typing import Callable, List


## Regex tutorial: 
##  - https://inbum.github.io/python/2018/04/26/python-regex/
##  - https://nachwon.github.io/regular-expressions/
##  - https://ko.wikipedia.org/wiki/%EB%8C%80%ED%95%9C%EB%AF%BC%EA%B5%AD%EC%9D%98_%EC%A0%84%ED%99%94%EB%B2%88%ED%98%B8_%EC%B2%B4%EA%B3%84


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
        supplementary_sentence_pattern = re.compile(r"^\s?[▶|\/][.]*", re.MULTILINE)

        texts = "\n".join(article_original)
        texts = supplementary_sentence_pattern.sub("", texts)
        
        return texts.split("\n")


    @staticmethod
    def replace_phone_number(article_original: List[str]) -> List[str]:
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
    def apply(article_original: List[str]) -> List[str]:
        ## Remove functions.
        texts = article_original
        
        texts = CleanNewspaperArticleBase.remove_reporter_info(texts)
        texts = CleanNewspaperArticleBase.remove_supplementary_sentence(texts)
        texts = CleanNewspaperArticleBase.remove_brack_sentence(texts)

        ## Replace functions.
        texts = CleanNewspaperArticleBase.replace_phone_number(texts)
        texts = CleanNewspaperArticleBase.replace_start_with_hyphen(texts)
        texts = CleanNewspaperArticleBase.replace_universal_apostrophe(texts)

        return texts


# class CleanLegalDocument():
    
#     def __init__(
#         self,
#     ):
#         super(CleanLegalDocument, self).__init__()
#         pass


# class CleanEditorialJournal():
    
#     def __init__(
#         self,
#     ):
#         super(CleanEditorialJournal, self).__init__()
#         pass


class CleanNewspaperArticle():
    
    def __init__(
        self,
    ):
        super(CleanNewspaperArticle, self).__init__()


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

        ## Skip -> cause error because of too many iteration steps... (too high complexity)
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


    def _cnews(self, article_original: List[str]) -> List[str]:
        """ """
        texts = []
        for sentence in article_original:
            texts.append(sentence)
        return texts


    def _cnews(self, article_original: List[str]) -> List[str]:
        """ """
        texts = []
        for sentence in article_original:
            texts.append(sentence)
        return texts


    def _cnews(self, article_original: List[str]) -> List[str]:
        """ """
        texts = []
        for sentence in article_original:
            texts.append(sentence)
        return texts


    def _cnews(self, article_original: List[str]) -> List[str]:
        """ """
        texts = []
        for sentence in article_original:
            texts.append(sentence)
        return texts


    def _cnews(self, article_original: List[str]) -> List[str]:
        """ """
        texts = []
        for sentence in article_original:
            texts.append(sentence)
        return texts


    def _cnews(self, article_original: List[str]) -> List[str]:
        """ """
        texts = []
        for sentence in article_original:
            texts.append(sentence)
        return texts


    def _cnews(self, article_original: List[str]) -> List[str]:
        """ """
        texts = []
        for sentence in article_original:
            texts.append(sentence)
        return texts


    def _cnews(self, article_original: List[str]) -> List[str]:
        """ """
        texts = []
        for sentence in article_original:
            texts.append(sentence)
        return texts


    def _cnews(self, article_original: List[str]) -> List[str]:
        """ """
        texts = []
        for sentence in article_original:
            texts.append(sentence)
        return texts


    def __call__(self, article_original: List[str], media_name: str) -> List[str]:
        texts = CleanNewspaperArticleBase.apply(article_original)

        media_name_to_function = {
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
        }
        func = media_name_to_function.get(media_name)
        if func != None:
            texts = func(texts)

        texts = [i.strip() for i in texts if i != ""]
        return texts
