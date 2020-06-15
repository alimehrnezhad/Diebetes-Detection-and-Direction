

from flask import Flask, render_template, request
app=Flask(__name__)



@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))
    if flask.request.method == 'POST':
        temperature = flask.request.form['temperature']
        humidity = flask.request.form['humidity']
        windspeed = flask.request.form['windspeed']
        input_variables = pd.DataFrame([[temperature, humidity, windspeed]],
                                       columns=['temperature', 'humidity', 'windspeed'],
                                       dtype=float)
        prediction = model.predict(input_variables)[0]
return flask.render_template('index.html',
        original_input={'Temperature':temperature,'Humidity':humidity,'Windspeed':windspeed},result=prediction,
                                     )
if __name__ == '__main__':
    app.run(debug = True)
