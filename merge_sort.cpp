#include <iostream>
#include <vector>

int merge(std::vector<int> &vec, int start, int mid, int end)
{
    std::vector<int> vec1(vec.begin() + start, vec.begin() + mid + 1);
    std::vector<int> vec2(vec.begin() + mid + 1, vec.begin() + end + 1);
    int p1 = 0, p1_sz = vec1.size();
    int p2 = 0, p2_sz = vec2.size();
    int p = start;
    int ret = 0;
    while (p1 < p1_sz && p2 < p2_sz) {
        if (vec1[p1] <= vec2[p2]) vec[p++] = vec1[p1++];
        else 
        {
            vec[p++] = vec2[p2++];
            ret += p1_sz - p1;
        }
    }
    if (p1 >= p1_sz) {
        while (p2 < p2_sz) 
        {
            vec[p++] = vec2[p2++];
            ret += p1_sz;
        }
    }
    else if (p2 >= p2_sz) {
        while (p1 < p1_sz) vec[p++] = vec1[p1++];
    }
    return ret;
}

int merge_sort(std::vector<int> &vec, int start, int end) {
    if (start == end) return 0;
    int mid = (start + end) / 2;
    int r1 = merge_sort(vec, start, mid);
    int r2 = merge_sort(vec, mid + 1, end);
    int r3 = merge(vec, start, mid, end);
    return r1 + r2 + r3;
}

int main()
{
    std::vector<int> vec{7, 6, 5, 4, 3, 2, 1};
    int ret = merge_sort(vec, 0, vec.size() - 1);
    for (auto &num : vec) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
    std::cout << ret << std::endl;
    return 0;
}