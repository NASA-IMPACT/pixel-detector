#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: class_download.py
# Author: JJ Miller
# Date: 2017-10-11
# Last Modified: 2018-05-17

'''
    This script uses curl to download the data files from CLASS
    the variable <path_file> is a text file that has the data directories for each CLASS order.
    You'll probably need to chamge some things around because I used storm names to organize everything.
    '''

import os
import time
import progressbar
import subprocess
from multiprocessing import Pool
import sys
import logging


def create_logger():
    # create logger for "Sample App"
    logger = logging.getLogger('class_download')
    logger.setLevel(logging.DEBUG)
    
    # create file handler which logs even debug messages
    fh = logging.FileHandler('logs/class_download.log', mode='w')
    fh.setLevel(logging.DEBUG)
    
    # create console handler with a higher log level
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    
    # create formatter and add it to the handlers
    file_formatter = logging.Formatter('[%(asctime)s] | %(levelname)8s: %(message)s ' +
                                       '(%(filename)s:%(lineno)s)',datefmt='%Y-%m-%d %H:%M:%S')
                                       std_out_formatter = logging.Formatter('[%(asctime)s] | %(levelname)8s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
                                       
                                       fh.setFormatter(file_formatter)
                                       ch.setFormatter(std_out_formatter)
                                       
                                       # add the handlers to the logger
                                       logger.addHandler(ch)
                                       logger.addHandler(fh)
                                       
    return logger


def rclone_download(dst_path,day_array,band_string = 'M3C01',year = '2018',hour = '00'):
    import os
    import subprocess
    '''rclone copy --include "*.nc" pubAWS:noaa-goes16/ABI-L1b-RadC/2018/315/ 
    /nas/rhome/mramasub/smoke_pixel_detector/data/input/sat_images/315
    '''

    for day in day_array:
        day = str (day)
        if band_string == '':
            band_str = '"*.nc"'
        else:    
            band_str = '"*'+band_string+'*.nc"'
        src_str = 'pubAWS:noaa-goes16/ABI-L1b-RadC/'+year+'/'+day+'/'+hour+'/'
        dst_str = os.path.join(dst_path,day,hour)

        if not os.path.exists(dst_str):
            os.makedirs(dst_str)

        
        print(band_str,src_str,dst_str)
        subprocess.call('rclone copy --include '+band_str+' '+src_str+' '+dst_str,shell=True)



def curl_download(bash_command):
    process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()


if __name__ == '__main__':
    logger = create_logger()
    
    for year in range(2017,2018):
        year = 2017
        basin = 'atlantic'
        path_file = '/rhome/mramasub/smoke_pixel_detector/raw_data/'
        
        with open(path_file, 'r') as t:
            paths = t.readlines()
    
        # constants
        base_path = '/rgroup/dsig/projects/intensity_estimation/data/raw_data/goes_data'
        username = 'anonymous'
        password = 'password'
        
        for path in paths:
            url = 'ftp.class.ngdc.noaa.gov'
            url_path = path.split(',')[1].strip()
            storm_name = path.split(',')[0].strip()
            logger.info('Downloading files for {0}.'.format(storm_name))
            
            # Use curl to list the files in the fpt
            bash_command = 'curl -sS ftp://{0}:{1}@{2}/{3}/001/ --tlsv1 -l'.format(username, password, url, url_path)
            process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            files, error = process.communicate()
            
            # CLASS uses 2 different urls. One with 'ncdc' and another wit 'ngdc'
            # We try 'ngdc' firjst and if there is an error we try 'ncdc'
            if error:
                logger.debug('Error: {0}'.format(error))
                url = 'ftp.class.ncdc.noaa.gov'
                logger.info('Using {0} to download data.'.format(url))
                bash_command = 'curl -sS ftp://{0}:{1}@{2}/{3}/001/ --tlsv1 -l'.format(username, password, url, url_path)
                process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                files, error = process.communicate()
        
            # We only want the band_03 and band_04 files
            # band_03 == water vapor
            # band_04 == IR
            try:
                files = files.split('\n')
                clean_files = []
                # get the year from the file name
                year = files[0].split('.')[1]
                # this is the local path where we store the data
                storm_path = os.path.join(base_path, 'data', basin, year, storm_name)
                if not os.path.exists(storm_path):
                    os.makedirs(storm_path)
                os.chdir(storm_path)
                for f in files:
                    if 'BAND_04' in f:
                        clean_files.append(f)
                    elif 'BAND_03' in f:
                        clean_files.append(f)
                    else:
                        continue
            
                logger.info('Storm_name: {0}; Year: {1}'.format(storm_name, year))
                bash_commands = []
                
                # download the files with curl
                for i, f in enumerate(clean_files):
                    file_name = f.strip()
                    bash_commands.append('curl -sS ftp://{0}:{1}@{2}/{3}/001/{4} --tlsv1 -O'.format(username, password, url, url_path, file_name))

                bar = progressbar.ProgressBar(redirect_stdout=True, max_value=len(bash_commands))
                p = Pool(4)

                for i, _ in enumerate(p.imap_unordered(curl_download, bash_commands), 1):
                    bar.update(i)
                        
            except Exception as e:
                logger.warning('Error: {0}'.format(e))
                logger.debug('Error downloading files for {0}'.format(storm_name))

