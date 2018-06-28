
directions = ["N","NE","E","SE","S","SO","O","NO"]

#
BEGINNING_CORNER = 0  # lower bound of N
directions_dict = {}
CORNER = 360 / 8
#directions_dict = {"N":(BEGINNING_CORNER,),"NE","E","SE","S","SO","O","NO"}
for d in directions:
    """
    directions_dict[d] = BEGINNING_CORNER
    BEGINNING_CORNER = CORNER + BEGINNING_CORNER
    """
    directions_dict[BEGINNING_CORNER] = d
    BEGINNING_CORNER = (CORNER + BEGINNING_CORNER) % 360


def get_dir(deg):
    #if (deg)
    deg = (deg + 360) % 360
    upper_bound = 45
    while deg > upper_bound:
        #print("dans la boucle")
        upper_bound = upper_bound + CORNER
    #print(upper_bound)
    return directions_dict[upper_bound - CORNER]

