# ucph_thesis
UCPH Master Thesis repository


for display your training result, you can use tensorboard 

```shell
tensorboard --logdir=<tensorboard_logdir> --port=<exposed_port>
```

if you are using ssh, you can bind the addess
```shell
ssh -L serverport:localhost:localport
```
where serverport should be equal to the exposed_port on tensorboard command
