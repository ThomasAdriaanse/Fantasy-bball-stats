# db_utils.py
import psycopg2
from psycopg2 import sql
from espn_api.basketball import League
import os
from datetime import datetime, timedelta



def is_player_in_database(player_name, table_name):
    db_params = get_connection_parameters()
    try:
        # Connect to the database using the connection parameters
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()

        # Prepare and execute the query with dynamic table name
        query = sql.SQL("SELECT EXISTS(SELECT 1 FROM {table} WHERE player_name = %s)").format(
            table=sql.Identifier(table_name)
        )
        cur.execute(query, (player_name,))

        # Fetch the result
        exists = cur.fetchone()[0]

        # Close the cursor and the connection
        cur.close()
        conn.close()

        return exists

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error while checking player in database: {error}")
        # In case of an error, close the connection
        if conn is not None:
            conn.close()
        return False

def create_table(cur, table_name, table_schema):
    create_table_query = sql.SQL("CREATE TABLE IF NOT EXISTS {table_name} ({table_schema})").format(
        table_name=sql.Identifier(table_name),
        table_schema=sql.SQL(table_schema)
    )
    cur.execute(create_table_query)

def drop_table(cur, table_name):
    drop_table_query = sql.SQL("DROP TABLE IF EXISTS {table_name}").format(
        table_name=sql.Identifier(table_name)
    )
    cur.execute(drop_table_query)


def insert_data_to_db(dataframe, table_name, column_names):
    db_params = get_connection_parameters()
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()


    insert_query = sql.SQL("INSERT INTO {table_name} ({fields}) VALUES ({values})").format(
        table_name=sql.Identifier(table_name),
        fields=sql.SQL(', ').join(map(sql.Identifier, column_names)),
        values=sql.SQL(', ').join(sql.Placeholder() * len(column_names))
    )

    for _, row in dataframe.iterrows():
        cur.execute(insert_query, [row[col] for col in column_names])
    
    conn.commit()
    cur.close()
    conn.close()


def get_db_cursor(db_params):
    conn = psycopg2.connect(**db_params)
    return conn, conn.cursor()

def get_db_connection():
    conn = psycopg2.connect(host=os.environ['DB_HOST'],
                            database=os.environ['DB_NAME'],
                            user=os.environ['DB_USER'],
                            password=os.environ['DB_PASS'])
    return conn

def get_connection_parameters():
    return {
        'host': os.environ['DB_HOST'],
        'database': os.environ['DB_NAME'],
        'user': os.environ['DB_USER'],
        'password': os.environ['DB_PASS']
    }

def range_of_current_week(date):
    #today = datetime.today()
    start_of_week = date - timedelta(days=date.weekday())
    end_of_week = start_of_week + timedelta(days=6)
    return start_of_week, end_of_week

def get_matchup_periods(league, current_matchup_period):
    return(league.settings.matchup_periods[str(current_matchup_period)])
