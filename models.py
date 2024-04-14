from django.db import models

# Create your models here.
class ckdModel(models.Model):

    Age=models.FloatField()
    DailyRate=models.FloatField()
    MonthlyIncome=models.FloatField()
    MonthlyRate=models.FloatField()
    TotalWorkingYears=models.FloatField()
    YearsAtCompany=models.FloatField()
    YearsInCurrentRole=models.FloatField()
    YearsWithCurrManager=models.FloatField()
