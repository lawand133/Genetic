from django import forms
from .models import pack



class packValues(forms.ModelForm):
    class Meta:
        model = pack
        fields = ['weight', 'value']

class limit(forms.Form):
    WeightLimit = forms.IntegerField()
    Iterations = forms.IntegerField()
    MutationProbabilty = forms.FloatField()