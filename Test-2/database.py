import psycopg2
import pandas as pd

#initisalise the postgrresql database
def db():
    conn = psycopg2.connect(database="pokemon", user="apoorv", password="1234", host="localhost", port=5433)
    curr = conn.cursor()
    return conn, curr

#load the dataset into a dataframe
def load_dataset():
    df = pd.read_excel('C:\Python35\Chillindo\pokemon.xlsx')
    return df

#insert values into table
def tables(conn, curr):
    create_sql = '''
    CREATE TABLE pokemon(_id sno, index integer, name varchar(25), type_1 varchar(10), type_2 varchar(10), total integer, hp integer, attack integer, defense integer, sp_atk integer, sp_def integer, speed integer, generation integer, legendary bool)
    '''
    curr.execute(create_sql)
    conn.commit()

#store values in postgresql db
def store_db(df, conn, curr):
    insert_sql = 'INSERT INTO pokemon VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'
    curr.executemany(insert_sql, df.reset_index().values)
    conn.commit()


if __name__ == '__main__':
    conn, curr = db()
    df = load_dataset()
    tables(conn, curr)
    store_db(df, conn, curr)
