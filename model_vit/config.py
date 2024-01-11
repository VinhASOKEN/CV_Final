Class_Info = {
    'num'  : 30,   
    'name' : [
        'ACBBank',
        'Agribank',
        'AnNinhNhanDan',
        'BacABank',
        'Baemin',
        'BambooAirways',
        'Be',
        'BIDV',
        'BoDoiBienPhong',
        'CanhSatCoDong',
        'CanhSatGiaoThong',
        'CanhSatNhanDan',
        'DanQuanTuVe',
        'GiaoHangNhanh',
        'Grab',
        'HaiQuan',
        'MBBank',
        'Now',
        'PacificAirlines',
        'PhongChayChuaChay',
        'PhongKhongKhongQuan',
        'QuanDoiKhac',
        'Shopee',
        'Techcombank',
        'TPBank',
        'Vietcombank',
        'Vietinbank',
        'VietjetAirlines',
        'VietnamAirlines',
        'ViettelPost'
    ]
}
Train_Config = {
    'path'            : "/data/disk2/vinhnguyen/Dino/train",
    'class'           : Class_Info,
    'image_size'      : (224, 224),
    'epoch'           : 29,
    'batch_size'      : 38,
    'learning_rate'   : 1e-5,
    'model_save_path' : '/data/disk2/vinhnguyen/Dino/model_vit/weight_vit',
    'load_checkpoint' : None
}

Valid_Config = {
    'path'            : "/data/disk2/vinhnguyen/Dino/valid",
    'class'           : Class_Info,
    'image_size'      : (224, 224),
    'batch_size'      : 38
}

Testing_Config = {
    'class'           : Class_Info,
    'image_size'      : (224, 224),
    'load_checkpoint' : '/data/disk2/vinhnguyen/Dino/model_vit/weight_vit/training_epoch_27.pth'
}