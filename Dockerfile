FROM ubuntu:latest

RUN apt-get update
RUN apt-get install -y unzip python3 python3-pip wget
RUN pip3 install numpy==1.14.5
RUN pip3 install scipy==1.1.0 gpflow==1.2.0 pyedflib==0.1.13 pywt==1.0.6 samplerate==0.1.0 tensorflow==1.9.0 scikit-image
RUN mkdir -p /narco/
ADD . /narco
RUN wget -P /narco/ml/ http://www.informaton.org/narco/ml/ac.zip 
RUN wget -P /narco/ml/ http://www.informaton.org/narco/ml/gp.zip
RUN wget -P /narco/ml/ www.informaton.org/narco/ml/scaling.zip 
RUN rm -r /narco/ml/ac /narco/ml/gp /narco/ml/scaling
RUN unzip /narco/ml/ac.zip -d /narco/ml/
RUN unzip /narco/ml/gp.zip -d /narco/ml/
RUN unzip /narco/ml/scaling.zip -d /narco/ml/
RUN ls /narco/
RUN ls /narco/ml/
RUN ls /narco/ml/ac
RUN ls /narco/ml/gp
RUN ls /narco/ml/scaling
RUN wget https://stanfordmedicine.box.com/shared/static/0lvvyaprzinzz7dult87t7hr96s2dnqq.edf
RUN mv 0lvvyaprzinzz7dult87t7hr96s2dnqq.edf /narco/CHP_040.edf
WORKDIR /narco
ENTRYPOINT ["/usr/bin/python3"]