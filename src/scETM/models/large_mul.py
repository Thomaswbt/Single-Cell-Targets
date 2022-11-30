
def large_mul(a,b):
    # c = torch.zeros_like(a)
    batch = min(1000,a.shape[0])
    i = 0
    while 1:
        if i + batch > a.shape[0]:
            a[i:,:] = a[i:,:] * b[i:,:]
            break
        a[i:i+batch,:] = a[i:i+batch,:]*b[i:i+batch,:]
        i += batch
    return a