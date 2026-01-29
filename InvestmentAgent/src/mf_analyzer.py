# mf_analyzer.py - Accurate & Modular Mutual Fund Support

from mftool import Mftool
import json
import re

class MutualFundManager:
    def __init__(self):
        self.mf = Mftool()
        print("Mutual Fund tool initialized — supports ALL Indian mutual funds")

    def _load_all_schemes(self):
        """Helper: Load full scheme list once per call"""
        try:
            return json.loads(self.mf.get_scheme_codes(as_json=True))
        except Exception as e:
            raise Exception(f"Failed to load scheme codes: {str(e)}")

    def search_by_amc(self, amc_name: str):
        """
        List all mutual funds from a specific AMC.
        Uses mftool's excellent get_available_schemes() method.
        Example: "ZERODHA", "ICICI", "HDFC"
        """
        amc_upper = amc_name.strip().upper()
        try:
            schemes = self.mf.get_available_schemes(amc_upper)
            if not schemes or not isinstance(schemes, dict):
                return {"error": f"No funds found for AMC '{amc_name}'"}
            
            results = [{"code": code, "name": name} for code, name in schemes.items()]
            return {
                "amc": amc_name.title(),
                "results": results,
                "count": len(results),
                "method": "AMC direct lookup"
            }
        except Exception as e:
            return {"error": f"AMC lookup failed for '{amc_name}': {str(e)}"}

    def search_by_name(self, name_query: str):
        """
        Search funds by partial or full name.
        Example: "nifty 250", "flexi cap", "gold etf"
        """
        query = name_query.strip()
        if not query:
            return {"error": "Name query cannot be empty"}

        try:
            all_schemes = self._load_all_schemes()
        except Exception as e:
            return {"error": str(e)}

        query_lower = query.lower()
        matches = []
        for code, name in all_schemes.items():
            if query_lower in name.lower():
                matches.append({"code": code, "name": name})

        if not matches:
            return {"error": f"No funds found matching name '{query}'"}

        matches.sort(key=lambda x: x["name"])
        results = matches[:30]

        return {
            "query": query,
            "results": results,
            "count": len(results),
            "method": "Name search"
        }

    def search_by_code(self, scheme_code: str):
        """
        Direct lookup by scheme code.
        Example: "152327", "152156"
        """
        code = scheme_code.strip()
        if not re.match(r'^\d{5,6}$', code):
            return {"error": f"Invalid scheme code format: '{code}'"}

        try:
            all_schemes = self._load_all_schemes()
        except Exception as e:
            return {"error": str(e)}

        if code not in all_schemes:
            return {"error": f"Scheme code '{code}' not found"}

        name = all_schemes[code]
        return {
            "code": code,
            "name": name,
            "results": [{"code": code, "name": name}],
            "count": 1,
            "method": "Direct code lookup"
        }

    def search_schemes(self, query: str):
        """
        Smart unified search — automatically picks the best method:
        - If contains 5-6 digit number → search_by_code
        - If contains known AMC keyword → search_by_amc
        - Otherwise → search_by_name
        """
        query = query.strip()
        original_query = query

        # 1. Direct scheme code detection
        code_match = re.search(r'\b(\d{5,6})\b', query)
        if code_match:
            return self.search_by_code(code_match.group(1))

        # 2. AMC keyword detection
        known_amcs = ["ZERODHA", "ICICI", "HDFC", "SBI", "AXIS", "KOTAK", 
                      "PARAG PARIKH", "PPFAS", "MOTILAL OSWAL", "UTI"]
        query_upper = query.upper()
        for amc in known_amcs:
            if amc in query_upper:
                return self.search_by_amc(amc)

        # 3. Fallback to name search
        return self.search_by_name(query)

    def get_latest_nav(self, scheme_code: str):
        """Get latest NAV for a scheme code"""
        try:
            quote = self.mf.get_scheme_quote(scheme_code)
            return {
                "code": scheme_code,
                "name": quote.get("scheme_name", "Unknown"),
                "nav": quote.get("nav"),
                "date": quote.get("date") or quote.get("last_updated", "N/A")
            }
        except Exception as e:
            return {"error": f"NAV fetch failed: {str(e)}"}

# Global instance
mf_manager = MutualFundManager()