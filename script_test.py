from api.schemas.predictions import ClinicalSummary, Severity

# This should crash:
try:
    ClinicalSummary(
        severity="urgent", summary="test", recommended_action="go",
        confidence_level="high", immediate_referral=False
    )
except Exception as e:
    print("Caught:", e)

# This should work:
s = ClinicalSummary(
    severity="urgent", summary="test", recommended_action="go",
    confidence_level="high", immediate_referral=True
)
print(s.severity, s.immediate_referral)