FROM python:3
RUN adduser pato

WORKDIR /home/pato

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install numpy

RUN pip3 install gunicorn
COPY app app

COPY entrypoint.sh entrypoint.sh


RUN chmod +x ./entrypoint.sh
RUN chown -R pato:pato ./

USER pato

EXPOSE 5004

ENTRYPOINT ["sh", "entrypoint.sh"]
