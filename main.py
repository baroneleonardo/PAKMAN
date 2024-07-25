from flask import Flask, request, jsonify, render_template, send_file
from qaliboo.pakman import PAKMAN 
from qaliboo.aux import stringa_per_stringa
from qaliboo.precomputed_functions import ScaledLiGenTot

app = Flask(__name__)

Baop = PM(n_initial_points=10, 
           n_iterations=30, 
           batch_size=5, 
           objective_func=ScaledLiGenTot,
           domain=ScaledLiGenTot, 
           lb=None, 
           ub=2.1, 
           nm=False,
           uniform_sample=True,
           save=False)

@app.route("/")
def home():
    #
    return stringa_per_stringa('ciao')

@app.route("/file-downloads/")
def file_downloads():
    return render_template('downloads.html')

@app.route("/return-file/")
def return_file():
    return send_file('/home/lbarone/QALIBOO/KISSGP.jpg')

@app.route("/aaa")
def naanan():
    return Baop.async_optimization(60,5) 



if __name__== "__main__":
    app.run(debug=True)

