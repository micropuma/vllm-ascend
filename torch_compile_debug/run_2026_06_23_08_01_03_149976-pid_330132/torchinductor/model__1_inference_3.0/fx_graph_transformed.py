class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[128, 512]"):
        # No stacktrace found for following nodes
        dbo_column_allgather_mlp: "f32[128, 512]" = torch.ops.vllm_ascend.dbo_column_allgather_mlp.default(arg0_1);  arg0_1 = None
        return (dbo_column_allgather_mlp,)
        