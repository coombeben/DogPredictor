FROM pytorch/pytorch

COPY requirements.txt /requirements.txt
WORKDIR /
RUN pip install -r requirements.txt

COPY . /