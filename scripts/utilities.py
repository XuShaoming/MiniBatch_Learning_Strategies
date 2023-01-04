import numpy as np
import torch

def get_NSE(Y, Y_hat):
    return 1 - np.sum(np.square(Y-Y_hat)) / np.sum(np.square(Y - np.mean(Y)))

def MSE(Y, Yhat, mask = None):
    if type(Y) == torch.Tensor:
        if mask is None:
            mask = torch.ones(Y.size())
            mask.to(Y.device)
        return torch.sum(torch.mul(torch.square(Y-Yhat), mask)) / (torch.sum(mask) + 1)
    
    if mask is None:
        mask = np.ones(Y.shape)
    return np.sum(np.multiply(np.square(Y-Yhat), mask)) / (np.sum(mask) + 1)

def print_notes(nlist):
    nlist = nlist.split('.')
    for nline in nlist:
        print(nline)
    return

def get_colors():
    plotly_colors='''
        aliceblue, antiquewhite, aqua, aquamarine, azure,
        beige, bisque, black, blanchedalmond, blue,
        blueviolet, brown, burlywood, cadetblue,
        chartreuse, chocolate, coral, cornflowerblue,
        cornsilk, crimson, cyan, darkblue, darkcyan,
        darkgoldenrod, darkgray, darkgrey, darkgreen,
        darkkhaki, darkmagenta, darkolivegreen, darkorange,
        darkorchid, darkred, darksalmon, darkseagreen,
        darkslateblue, darkslategray, darkslategrey,
        darkturquoise, darkviolet, deeppink, deepskyblue,
        dimgray, dimgrey, dodgerblue, firebrick,
        floralwhite, forestgreen, fuchsia, gainsboro,
        ghostwhite, gold, goldenrod, gray, grey, green,
        greenyellow, honeydew, hotpink, indianred, indigo,
        ivory, khaki, lavender, lavenderblush, lawngreen,
        lemonchiffon, lightblue, lightcoral, lightcyan,
        lightgoldenrodyellow, lightgray, lightgrey,
        lightgreen, lightpink, lightsalmon, lightseagreen,
        lightskyblue, lightslategray, lightslategrey,
        lightsteelblue, lightyellow, lime, limegreen,
        linen, magenta, maroon, mediumaquamarine,
        mediumblue, mediumorchid, mediumpurple,
        mediumseagreen, mediumslateblue, mediumspringgreen,
        mediumturquoise, mediumvioletred, midnightblue,
        mintcream, mistyrose, moccasin, navajowhite, navy,
        oldlace, olive, olivedrab, orange, orangered,
        orchid, palegoldenrod, palegreen, paleturquoise,
        palevioletred, papayawhip, peachpuff, peru, pink,
        plum, powderblue, purple, red, rosybrown,
        royalblue, saddlebrown, salmon, sandybrown,
        seagreen, seashell, sienna, silver, skyblue,
        slateblue, slategray, slategrey, snow, springgreen,
        steelblue, tan, teal, thistle, tomato, turquoise,
        violet, wheat, white, whitesmoke, yellow,
        yellowgreen
        '''
    colors=plotly_colors.split(',')
    colors=[l.replace('\n','') for l in colors]
    colors=[l.replace(' ','') for l in colors]
    colors = [c for c in colors if "dark" in c or "deep" in c]
    colors = np.unique(colors)
    colors = np.random.permutation(colors)
    return colors

'''
This code block inverse_transform predictions to the original scale.
'''
def inverse_transform_sp(pred_sp,y_sp, scaler_y, n_classes):
    pred_sp = np.squeeze(pred_sp)
    pred_shape = pred_sp.shape
    pred_sp= pred_sp.reshape(-1,n_classes)
    pred_sp = scaler_y.inverse_transform(pred_sp).reshape(pred_shape)
    
    y_ori =  np.squeeze(y_sp).reshape(-1,n_classes)
    y_ori = scaler_y.inverse_transform(y_ori).reshape(pred_shape)
    
    return pred_sp, y_ori