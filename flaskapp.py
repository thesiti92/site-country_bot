from flask import Flask, render_template, flash, request

from gentxt import gentxt
from wtforms import Form, IntegerField, StringField, validators
import os

class ParamsForm(Form):
    primetext = StringField('Primetext', [validators.DataRequired()])
    length = IntegerField('Length', default=50)
    sample = IntegerField('Sample', default=1)
    seed = IntegerField('Random Seed', default=123)

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    form = ParamsForm(request.form)
    if request.method == 'POST' and form.validate() and form.primetext.data != "":
        flash('Generating Text')
        result = gentxt(primetext=form.primetext.data, length=form.length.data, sample=form.sample.data, seed=form.seed.data)
        if result == "unknown":
            return render_template('index.html', form=form, result="The given primetext wasn't recognized")
        return render_template('index.html', form=form, result=result)
    return render_template('index.html', form=form)

if __name__ == "__main__":
    app.secret_key = 'super secret key'
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
