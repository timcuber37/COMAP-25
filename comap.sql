create database comap;
use comap;

create table programs
(
	id int not null primary key,
	sport varchar(50) null,
    discipline varchar(50) null,
    sportscode varchar(10) null,
    govbody varchar(30) null,
    num1896 int null,
    num1900 int null,
    num1904 int null,
    num1906 int null,
    num1908 int null,
    num1912 int null,
    num1920 int null,
    num1924 int null,
	num1928 int null,
    num1932 int null,
    num1936 int null,
    num1948 int null,
    num1952 int null,
    num1956 int null,
    num1960 int null,
    num1964 int null,
    num1968 int null,
    num1972 int null,
    num1976 int null,
    num1980 int null,
    num1984 int null,
    num1988 int null,
    num1992 int null,
    num1996 int null,
    num2000 int null,
    num2004 int null,
    num2008 int null,
    num2012 int null,
    num2016 int null,
    num2020 int null,
    num2024 int null
);

create table athletes
(
	id int not null primary key,
	name varchar(100) null,
    sex char null,
    team varchar(100) null,
    noc varchar(10) null,
    year int null,
    city varchar(50) null,
    sport varchar(50) null,
    event varchar(100) null,
    medal varchar(20) null
);

select* from athletes
where year = 2024 && medal != 'No medal' && team = 'United States';

select*from programs where sport is null or
discipline is null or
sportscode is null or
govbody is null or 
num1896 is null or
num1900 is null or
num1904 is null or
num1906 is null or
num1908 is null or
num1912 is null or
num1920 is null or
num1924 is null or
num1928 is null or
num1932 is null or
num1936 is null or
num1948 is null or
num1952 is null or
num1956 is null or
num1960 is null or
num1964 is null or
num1968 is null or
num1972 is null or
num1976 is null or
num1980 is null or
num1984 is null or
num1988 is null or
num1992 is null or
num1996 is null or
num2000 is null or
num2004 is null or
num2008 is null or
num2012 is null or
num2016 is null or
num2020 is null or
num2024 is null;


