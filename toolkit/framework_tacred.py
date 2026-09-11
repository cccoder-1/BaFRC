from toolkit.framework import FewShotREFramework, FewShotREModel


class FewShotREFrameworkTacred(FewShotREFramework):
    """
    FS-TACRED-style framework wrapper.
    Uses target_micro_f1 for checkpoint selection.
    """

    def train(self, *args, **kwargs):
        kwargs.setdefault("primary_metric", "target_micro_f1")
        return super().train(*args, **kwargs)
