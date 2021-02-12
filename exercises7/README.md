# MakeArt
Aim of this project is to create web application to create art based on two provided images (one to transform, one to apply style from).
It uses pretrained Tensorflow Hub model arbitrary-image-stylization-v1-256 to perform transformation.
Project is prepared as Docker image, but you can run it locally (however it might be more difficult).

To build this project you have to run this command:  
``` docker build . -t makeart ```

To run it:  
``` docker run -p 8000:8000 makeart ```

If you want to run it locally, first you have to install all the dependencies:
``` pip install -r requirements ```

Then run unicorn with server:
``` "uvicorn app:app --host 0.0.0.0```


Application should be available at ```http://localhost:8000```

Please note that it might take about a minute for the results  

Here is example output, based on famous Guernica painting made by Pablo Picasso:  

