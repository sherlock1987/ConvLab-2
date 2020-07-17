
a = [[['Inform', 'Train', 'Choice', 'over 1'], ['Inform', 'Train', 'Choice', '000'], ['Request', 'Train', 'Depart', '?']],
[['Inform', 'Train', 'Depart', 'birmingham new street']],
[['Request', 'Train', 'Day', '?']],
[['Inform', 'Train', 'Day', 'wednesday']],
[['Inform', 'Train', 'Arrive', '20:23'], ['Inform', 'Train', 'Day', 'Wednesday'], ['Inform', 'Train', 'Depart', 'birmingham new street'], ['Inform', 'Train', 'Leave', '17:40']],
[['Inform', 'Train', 'People', '5']],
[['OfferBooked', 'Train', 'Ref', 'A9NHSO9Y']],
[['Inform', 'Hotel', 'Internet', 'yes'], ['Inform', 'Hotel', 'Stars', '4']],
[['Recommend', 'Hotel', 'Name', 'the cambridge belfry']],
[['Inform', 'Hotel', 'Day', 'wednesday'], ['Inform', 'Hotel', 'People', '5'], ['Inform', 'Hotel', 'Stay', '5']],
[['Book', 'Booking', 'Ref', '5NAWGJDC']],
[['thank', 'general', 'none', 'none']],
[['bye', 'general', 'none', 'none']]]

print(a)
for i in range(len(a)):
    print()
    for j in range(len(a[i])):
        one = a[i][j]
        left = "_".join(one[:-1])
        right = one[-1]
        whole = ": ".join([left, right])
        print(whole)