FROM puckel/docker-airflow
WORKDIR /airflow
COPY dags/* /usr/local/airflow/dags/
RUN ls -l /usr/local/airflow/dags/
RUN pip3 install boto3