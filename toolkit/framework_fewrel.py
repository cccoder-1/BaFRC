from toolkit.framework import FewShotREFramework, FewShotREModel


class FewShotREFrameworkFewRel(FewShotREFramework):
    """
    FewRel-style framework wrapper.
    Keeps legacy checkpoint selection (acc + macro_f1).
    """

    def train(self, *args, **kwargs):
        kwargs.setdefault("primary_metric", "legacy")
        return super().train(*args, **kwargs)
